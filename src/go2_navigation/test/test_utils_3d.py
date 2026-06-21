"""3D 导航工具函数的单元测试。"""

import numpy as np
import pytest

from go2_navigation.utils import (
    voxelize_ground,
    extract_keypoints,
    build_nav_graph,
    find_nearest_keypoint,
    dijkstra_search,
    graph_path_to_world,
    smooth_path_3d,
)


# ── voxelize_ground ───────────────────────────────────────────


class TestVoxelizeGround:
    def test_flat_ground_produces_voxels(self, flat_ground_points):
        """平地点云应产生多个有效体素。"""
        centers, heights = voxelize_ground(flat_ground_points, 0.2, 3, 0.5)
        assert len(centers) > 0
        assert len(centers) == len(heights)
        # 平地高度应接近 0.1m
        assert np.abs(heights.mean() - 0.1) < 0.05

    def test_filters_high_points(self):
        """超过 ground_z_max 的点应被排除。"""
        pts = np.array([[0.01, 0.01, 0.1]] * 5 + [[0.01, 0.01, 1.0]] * 5)
        centers, heights = voxelize_ground(pts, 0.2, 3, 0.5)
        assert len(centers) == 1  # 只有 z=0.1 的体素
        assert abs(heights[0] - 0.1) < 0.01

    def test_min_points_filter(self):
        """点数不足的体素应被排除。"""
        pts = np.array([[0.01, 0.01, 0.1], [0.02, 0.02, 0.1]])  # 只有 2 个点
        centers, heights = voxelize_ground(pts, 0.2, 3, 0.5)
        assert len(centers) == 0

    def test_empty_input(self):
        """空输入应返回空数组。"""
        pts = np.empty((0, 3), dtype=np.float32)
        centers, heights = voxelize_ground(pts, 0.2, 3, 0.5)
        assert len(centers) == 0
        assert len(heights) == 0

    def test_multiple_voxels(self):
        """相距较远的点应产生多个体素。"""
        pts1 = np.random.rand(5, 3).astype(np.float32) * 0.1
        pts1[:, 2] = 0.1
        pts2 = np.random.rand(5, 3).astype(np.float32) * 0.1 + np.array([2.0, 0, 0])
        pts2[:, 2] = 0.1
        pts = np.vstack([pts1, pts2])
        centers, heights = voxelize_ground(pts, 0.2, 3, 0.5)
        assert len(centers) == 2


# ── extract_keypoints ─────────────────────────────────────────


class TestExtractKeypoints:
    def test_downsampling_reduces_count(self, flat_ground_points):
        """降采样应减少关键点数量。"""
        centers, heights = voxelize_ground(flat_ground_points, 0.2, 3, 0.5)
        kps, kp_heights = extract_keypoints(centers, heights, 0.5)
        assert len(kps) <= len(centers)
        assert len(kps) > 0

    def test_empty_input(self):
        """空输入应返回空数组。"""
        kps, heights = extract_keypoints(
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.2,
        )
        assert len(kps) == 0


# ── build_nav_graph ───────────────────────────────────────────


class TestBuildNavGraph:
    def test_flat_grid_connectivity(self):
        """平地网格上的关键点应全部连通。"""
        # 3x3 网格，z=0，间距 0.3m
        kps = np.array(
            [[i * 0.3, j * 0.3, 0.0] for i in range(3) for j in range(3)],
            dtype=np.float32,
        )
        heights = np.zeros(9, dtype=np.float32)
        graph = build_nav_graph(kps, heights, 0.5, 0.2, 45.0)
        # 中心点应与 8 个邻居相连
        center_idx = 4  # (0.3, 0.3, 0.0)
        assert len(graph.adjacency[center_idx]) == 8
        assert graph.tree is not None
        assert len(graph.traversable) == 9

    def test_height_filter_removes_edges(self):
        """高差超过阈值的边应被移除。"""
        kps = np.array([[0, 0, 0], [0.3, 0, 0.5]], dtype=np.float32)
        heights = np.array([0.0, 0.5], dtype=np.float32)
        graph = build_nav_graph(kps, heights, 0.5, 0.2, 45.0)
        assert len(graph.adjacency[0]) == 0  # 高差 0.5 > 0.2，无连接

    def test_slope_filter_removes_steep_edges(self):
        """超过坡度阈值的边应被移除。"""
        kps = np.array([[0, 0, 0], [0.1, 0, 0.5]], dtype=np.float32)
        heights = np.array([0.0, 0.5], dtype=np.float32)
        graph = build_nav_graph(kps, heights, 0.5, 1.0, 30.0)
        assert len(graph.adjacency[0]) == 0  # 坡度 > 30°

    def test_empty_input(self):
        """空输入应返回空图。"""
        graph = build_nav_graph(
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.5,
            0.2,
            45.0,
        )
        assert len(graph.keypoints) == 0
        assert len(graph.adjacency) == 0


# ── find_nearest_keypoint ─────────────────────────────────────


class TestFindNearestKeypoint:
    def test_finds_closest(self):
        """应返回最近关键点的索引。"""
        kps = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        graph = build_nav_graph(kps, np.zeros(3), 2.0, 1.0, 45.0)
        idx = find_nearest_keypoint(graph.tree, (0.1, 0.0, 0.0))
        assert idx == 0

    def test_exact_match(self):
        """精确位置应返回对应索引。"""
        kps = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        graph = build_nav_graph(kps, np.zeros(2), 2.0, 1.0, 45.0)
        idx = find_nearest_keypoint(graph.tree, (1.0, 0.0, 0.0))
        assert idx == 1


# ── dijkstra_search ───────────────────────────────────────────


class TestDijkstraSearch:
    def test_linear_path(self):
        """线性图应找到 0→1→2。"""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 1.0)], 2: [(1, 1.0)]}
        path = dijkstra_search(adj, 0, 2)
        assert path == [0, 1, 2]

    def test_prefers_shorter_path(self):
        """应选择更短的路径。"""
        adj = {
            0: [(1, 1.0), (2, 10.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(0, 10.0), (1, 1.0)],
        }
        path = dijkstra_search(adj, 0, 2)
        assert path == [0, 1, 2]

    def test_disconnected_returns_none(self):
        """不连通的图应返回 None。"""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: []}
        path = dijkstra_search(adj, 0, 2)
        assert path is None

    def test_same_start_goal(self):
        """起点等于终点应返回单元素路径。"""
        adj = {0: [(1, 1.0)]}
        path = dijkstra_search(adj, 0, 0)
        assert path == [0]


# ── graph_path_to_world ───────────────────────────────────────


class TestGraphPathToWorld:
    def test_single_point(self):
        """单点路径应返回该点。"""
        kps = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = graph_path_to_world([0], kps, 5)
        assert len(result) == 1
        assert result[0] == (1.0, 2.0, 3.0)

    def test_interpolation(self):
        """两点之间应有插值点。"""
        kps = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        result = graph_path_to_world([0, 1], kps, 5)
        # 应有 5 个插值点 + 1 个终点 = 6 个点
        assert len(result) == 6
        # 第一个点应是起点
        assert abs(result[0][0]) < 0.01
        # 最后一个点应是终点
        assert abs(result[-1][0] - 1.0) < 0.01

    def test_empty_path(self):
        """空路径应返回空列表。"""
        kps = np.array([[0, 0, 0]], dtype=np.float32)
        result = graph_path_to_world([], kps, 5)
        assert len(result) == 0


# ── smooth_path_3d ────────────────────────────────────────────


class TestSmoothPath3D:
    def test_short_path_unchanged(self):
        """2 点以下的路径不应改变。"""
        path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        result = smooth_path_3d(path, 3, 0.5)
        assert len(result) == 2
        assert result == path

    def test_smoothing_reduces_roughness(self):
        """平滑应减少路径的锯齿。"""
        # Z 形路径
        path = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.5, 0.0),
            (2.0, 0.0, 0.0),
        ]
        result = smooth_path_3d(path, 3, 0.5)
        assert len(result) == len(path)
        # 端点不变
        assert result[0] == path[0]
        assert result[-1] == path[-1]
        # 中间点的 Y 值应比原来小（被平滑了）
        mid_y_orig = path[1][1]
        mid_y_smooth = result[1][1]
        assert mid_y_smooth < mid_y_orig

    def test_height_weight_affects_z(self):
        """height_weight 应影响 Z 方向平滑程度。"""
        path = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
        ]
        # 全权重平滑
        result_full = smooth_path_3d(path, 3, 1.0)
        # 无 Z 平滑
        result_none = smooth_path_3d(path, 3, 0.0)
        # 全权重的中间点 Z 应更接近均值
        assert abs(result_full[1][2] - 0.5) < abs(result_none[1][2] - 0.5)
