"""3D 运动控制器单元测试。"""

import math
import numpy as np
import pytest

from go2_navigation.motion_controller import (
    euler_from_quaternion,
    pure_pursuit_control,
    smooth_velocity,
    find_look_ahead_point,
)


class TestEulerFromQuaternion:
    def test_identity_quaternion(self):
        yaw = euler_from_quaternion((0.0, 0.0, 0.0, 1.0))
        assert abs(yaw) < 1e-6

    def test_90_degree_rotation(self):
        yaw = euler_from_quaternion(
            (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        )
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_180_degree_rotation(self):
        yaw = euler_from_quaternion((0.0, 0.0, 1.0, 0.0))
        assert abs(abs(yaw) - math.pi) < 1e-6


class TestPurePursuitControl3D:
    """测试 3D 版 Pure Pursuit 控制律。"""

    def test_target_straight_ahead_flat(self):
        """平地上目标在正前方应产生正线速度。"""
        vx, vyaw = pure_pursuit_control(0, 0, 0, 0, 1.0, 0.0, 0.0)
        assert vx > 0
        assert abs(vyaw) < 0.1

    def test_target_to_the_left(self):
        """目标在左侧应产生正角速度。"""
        _, vyaw = pure_pursuit_control(0, 0, 0, 0, 0.5, 0.5, 0.0)
        assert vyaw > 0

    def test_target_to_the_right(self):
        """目标在右侧应产生负角速度。"""
        _, vyaw = pure_pursuit_control(0, 0, 0, 0, 0.5, -0.5, 0.0)
        assert vyaw < 0

    def test_target_at_same_position(self):
        """目标在同位置应返回零速度。"""
        vx, vyaw = pure_pursuit_control(0, 0, 0, 0, 0.0, 0.0, 0.0)
        assert abs(vx) < 0.01
        assert abs(vyaw) < 0.01

    def test_slope_reduces_speed(self):
        """上坡应降低线速度。"""
        vx_flat, _ = pure_pursuit_control(
            0, 0, 0, 0, 2.0, 0.0, 0.0, max_lin=0.5, slope_speed_factor=0.5
        )
        vx_slope, _ = pure_pursuit_control(
            0, 0, 0, 0, 2.0, 0.0, 1.0, max_lin=0.5, slope_speed_factor=0.5
        )
        assert vx_slope < vx_flat

    def test_flat_slope_no_speed_change(self):
        """平地不应有坡度减速。"""
        vx, _ = pure_pursuit_control(
            0, 0, 0, 0, 1.0, 0.0, 0.0, max_lin=0.5, slope_speed_factor=0.5
        )
        assert vx == pytest.approx(0.5, abs=0.05)

    def test_max_speed_clamp(self):
        """速度应被限制在最大值内。"""
        vx, vyaw = pure_pursuit_control(
            0, 0, 0, 0, 10.0, 0.0, 0.0, max_lin=0.2, max_ang=0.5
        )
        assert vx <= 0.2
        assert abs(vyaw) <= 0.5

    def test_speed_never_below_30_percent(self):
        """即使在最大坡度下，速度也不应低于 30%。"""
        vx, _ = pure_pursuit_control(
            0, 0, 0, 0,
            1.0, 0.0, 10.0,  # 很大的高差
            max_lin=0.5,
            max_slope_angle=30.0,
            slope_speed_factor=0.5,
        )
        assert vx >= 0.5 * 0.3  # 至少 30%


class TestSmoothVelocity:
    def test_no_change_needed(self):
        result = smooth_velocity(0.2, 0.2, 0.1)
        assert result == pytest.approx(0.2)

    def test_increase_within_limit(self):
        result = smooth_velocity(0.2, 0.0, 1.0, max_acc=0.5)
        assert result == pytest.approx(0.2)

    def test_increase_exceeds_limit(self):
        result = smooth_velocity(1.0, 0.0, 0.1, max_acc=0.3)
        assert result == pytest.approx(0.03)

    def test_decrease_exceeds_limit(self):
        result = smooth_velocity(0.0, 1.0, 0.1, max_acc=0.3)
        assert result == pytest.approx(0.97)


class TestFindLookAheadPoint3D:
    def test_basic(self):
        """基本 3D 前视点查找。"""
        path = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        target = find_look_ahead_point(path, 0.0, 0.0, 0.0, 0, look_ahead=0.3)
        assert target is not None
        assert target[0] >= 0.3

    def test_3d_distance_used(self):
        """应使用 3D 距离而非仅 XY。"""
        path = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.5),  # XY 距离小但 3D 距离大
            (1.0, 0.0, 0.5),
        ]
        target = find_look_ahead_point(path, 0.0, 0.0, 0.0, 0, look_ahead=0.3)
        assert target is not None
        # 第二个点的 3D 距离 = sqrt(0.01 + 0.25) ≈ 0.51 > 0.3
        assert target[0] == pytest.approx(0.1, abs=0.01)

    def test_at_end_of_path(self):
        """在路径末端应返回最后一个点。"""
        path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.5)]
        target = find_look_ahead_point(path, 0.9, 0.0, 0.0, 1, look_ahead=0.3)
        assert target == (1.0, 0.0, 0.5)

    def test_empty_path(self):
        target = find_look_ahead_point([], 0.0, 0.0, 0.0, 0)
        assert target is None
