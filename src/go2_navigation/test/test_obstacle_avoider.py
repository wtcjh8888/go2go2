"""3D 避障模块单元测试。"""

import math
import numpy as np
import pytest

from go2_navigation.obstacle_avoider import (
    build_vfh_histogram,
    find_best_direction,
    check_danger_ahead_3d,
    filter_fov_points_3d,
    classify_terrain,
    compute_avoidance_cmd,
)


class TestBuildVFHHistogram:
    """测试 VFH 直方图构建。"""

    def test_empty_pointcloud(self):
        """空输入应返回全 detection_range。"""
        histogram = build_vfh_histogram(np.array([]), np.array([]))
        assert len(histogram) == 72
        assert np.all(histogram == 2.0)

    def test_obstacle_directly_ahead(self):
        """正前方障碍物应反映在扇区 0。"""
        dists = np.array([0.5])
        angles = np.array([0.0])
        histogram = build_vfh_histogram(dists, angles)
        assert histogram[0] == pytest.approx(0.5)

    def test_obstacle_to_the_right(self):
        """右侧 45° 障碍物。"""
        dists = np.array([1.0])
        angles = np.array([-math.radians(45)])
        histogram = build_vfh_histogram(dists, angles)
        sector = int(315.0 / 5.0) % 72  # -45° → 315° → sector 63
        assert histogram[sector] == pytest.approx(1.0)

    def test_multiple_obstacles_same_sector(self):
        """同一扇区多个障碍物取最近距离。"""
        dists = np.array([1.0, 0.3])
        angles = np.array([0.0, 0.0])
        histogram = build_vfh_histogram(dists, angles)
        assert histogram[0] == pytest.approx(0.3)

    def test_default_range(self):
        """没有障碍物的扇区保持 detection_range。"""
        dists = np.array([0.5])
        angles = np.array([0.0])
        histogram = build_vfh_histogram(dists, angles, detection_range=2.0)
        assert histogram[0] == pytest.approx(0.5)
        assert histogram[1] == pytest.approx(2.0)


class TestFindBestDirection:
    """测试最佳方向查找。"""

    def test_all_clear(self):
        """所有扇区都通畅时应返回 0°（正前方）。"""
        histogram = np.full(72, 3.0)
        angle = find_best_direction(histogram, warning_dist=0.6)
        assert angle is not None
        assert abs(angle) < 10.0 or abs(angle - 360.0) < 10.0

    def test_obstacle_ahead_go_left(self):
        """正前方有障碍时应偏向左侧。"""
        histogram = np.full(72, 3.0)
        histogram[0] = 0.2
        angle = find_best_direction(histogram, warning_dist=0.6)
        assert angle is not None
        assert angle > 0

    def test_obstacle_ahead_go_right(self):
        """正前方有障碍且左侧也有时应偏向右侧。"""
        histogram = np.full(72, 3.0)
        histogram[0] = 0.2
        for i in range(1, 10):
            histogram[i] = 0.2
        angle = find_best_direction(histogram, warning_dist=0.6)
        assert angle is not None
        assert angle > 300.0 or angle < 0

    def test_all_blocked(self):
        """所有方向都被阻挡时应返回 None。"""
        histogram = np.full(72, 0.2)
        angle = find_best_direction(histogram, warning_dist=0.6)
        assert angle is None


class TestCheckDangerAhead3D:
    """测试 3D 前方危险检测。"""

    def test_danger_in_front(self):
        """正前方危险距离内的障碍物。"""
        points = np.array([[0.2, 0.0, 0.0]])
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3) is True

    def test_no_danger(self):
        """远处无障碍物。"""
        points = np.array([[5.0, 5.0, 0.0]])
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3) is False

    def test_outside_fov(self):
        """FOV 外的障碍物不检测。"""
        points = np.array([[0.0, 2.0, 0.0]])
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3, fov_angle=120.0) is False

    def test_empty_points(self):
        """空点云。"""
        points = np.array([]).reshape(0, 3)
        assert check_danger_ahead_3d(points, robot_z=0.0) is False

    def test_boundary_fov(self):
        """刚好在 FOV 边界上的点。"""
        angle = math.radians(59.0)
        x = 0.2 * math.cos(angle)
        y = 0.2 * math.sin(angle)
        points = np.array([[x, y, 0.0]])
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3, fov_angle=120.0) is True

    def test_height_filter_ignores_ground(self):
        """地面以下的点应被忽略。"""
        points = np.array([[0.2, 0.0, -0.5]])  # 远低于机器人
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3, height_band=0.5, ground_clearance=0.05) is False

    def test_height_filter_ignores_overhead(self):
        """头顶以上的点应被忽略。"""
        points = np.array([[0.2, 0.0, 2.0]])  # 远高于机器人
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3, height_band=0.5) is False

    def test_height_filter_passes_valid(self):
        """机器人高度附近的点应被检测。"""
        points = np.array([[0.2, 0.0, 0.1]])  # 在 height_band 内
        assert check_danger_ahead_3d(points, robot_z=0.0, danger_dist=0.3, height_band=0.5) is True


class TestFilterFOVPoints3D:
    """测试 3D FOV 点云过滤。"""

    def test_filter_range(self):
        """超出范围的点应被过滤。"""
        points = np.array([
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        dists, angles = filter_fov_points_3d(points, robot_z=0.0, detection_range=2.0)
        assert len(dists) == 1
        assert dists[0] == pytest.approx(0.5)

    def test_filter_fov(self):
        """FOV 外的点应被过滤。"""
        points = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ])
        dists, angles = filter_fov_points_3d(points, robot_z=0.0, fov_angle=120.0)
        assert len(dists) == 1

    def test_empty_input(self):
        """空输入。"""
        points = np.array([]).reshape(0, 3)
        dists, angles = filter_fov_points_3d(points, robot_z=0.0)
        assert len(dists) == 0
        assert len(angles) == 0

    def test_height_filtering(self):
        """应过滤掉不在高度带内的点。"""
        points = np.array([
            [0.5, 0.0, 0.0],    # 有效高度
            [0.5, 0.0, -0.5],   # 地面以下
            [0.5, 0.0, 2.0],    # 头顶以上
        ])
        dists, angles = filter_fov_points_3d(points, robot_z=0.0, height_band=0.5, ground_clearance=0.05)
        assert len(dists) == 1


class TestClassifyTerrain:
    """测试地形分类。"""

    def test_flat_ground(self):
        """平地点云应返回非坡道。"""
        # 生成足够多的平地点（至少 10 个）
        pts = []
        for x in np.arange(0.2, 1.5, 0.15):
            for y in np.arange(0.0, 0.6, 0.3):
                pts.append([x, y, 0.0])
        points = np.array(pts)
        is_slope, angle = classify_terrain(points, robot_z=0.0, slope_threshold=30.0)
        assert is_slope is False
        assert angle < 30.0

    def test_slope_detected(self):
        """明显坡度应被检测。"""
        # 生成足够多的坡道点（至少 10 个），z 随 x 线性增加
        pts = []
        for x in np.arange(0.2, 1.5, 0.15):
            for y in np.arange(0.0, 0.6, 0.3):
                z = (x - 0.2) * 0.8  # 坡度约 38.7°
                pts.append([x, y, z])
        points = np.array(pts)
        is_slope, angle = classify_terrain(points, robot_z=0.0, slope_threshold=30.0)
        assert is_slope is True
        assert angle >= 30.0

    def test_empty_points(self):
        """空输入应返回非坡道。"""
        points = np.array([]).reshape(0, 3)
        is_slope, angle = classify_terrain(points, robot_z=0.0)
        assert is_slope is False
        assert angle == 0.0


class TestComputeAvoidanceCmd:
    """测试避障速度计算。"""

    def test_straight_ahead(self):
        """0° 方向应保持原始速度。"""
        vx, vyaw = compute_avoidance_cmd(0.0, raw_vx=0.2)
        assert vx > 0
        assert abs(vyaw) < 0.1

    def test_turn_left(self):
        """正角度应产生正角速度。"""
        vx, vyaw = compute_avoidance_cmd(30.0, raw_vx=0.2)
        assert vyaw > 0

    def test_turn_right(self):
        """负角度应产生负角速度。"""
        vx, vyaw = compute_avoidance_cmd(-30.0, raw_vx=0.2)
        assert vyaw < 0

    def test_large_angle_reduces_speed(self):
        """大角度时线速度应降低。"""
        vx_small, _ = compute_avoidance_cmd(10.0, raw_vx=0.2)
        vx_large, _ = compute_avoidance_cmd(80.0, raw_vx=0.2)
        assert vx_large < vx_small

    def test_speed_clamp(self):
        """速度应被限制在最大值内。"""
        _, vyaw = compute_avoidance_cmd(90.0, max_ang=0.5)
        assert abs(vyaw) <= 0.5
