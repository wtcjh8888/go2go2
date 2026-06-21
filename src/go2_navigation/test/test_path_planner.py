"""3D 路径规划器单元测试。"""

import numpy as np
import pytest

from go2_navigation.utils import (
    dijkstra_search,
    graph_path_to_world,
    smooth_path_3d,
    build_nav_graph,
)


class TestDijkstraSearch:
    """测试 Dijkstra 搜索算法。"""

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

    def test_complex_graph(self):
        """复杂图应找到最优路径。"""
        adj = {
            0: [(1, 2.0), (2, 5.0)],
            1: [(0, 2.0), (3, 3.0)],
            2: [(0, 5.0), (3, 1.0)],
            3: [(1, 3.0), (2, 1.0), (4, 2.0)],
            4: [(3, 2.0)],
        }
        path = dijkstra_search(adj, 0, 4)
        assert path is not None
        assert path[0] == 0
        assert path[-1] == 4
        # 最优路径: 0→1→3→4 (cost=7) 优于 0→2→3→4 (cost=8)
        assert path == [0, 1, 3, 4]


class TestGraphPathToWorld:
    """测试图路径转世界坐标。"""

    def test_single_point(self):
        """单点路径应返回该点。"""
        kps = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = graph_path_to_world([0], kps, 5)
        assert len(result) == 1
        assert result[0] == (1.0, 2.0, 3.0)

    def test_interpolation_count(self):
        """两点之间应有正确数量的插值点。"""
        kps = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        result = graph_path_to_world([0, 1], kps, 5)
        assert len(result) == 6  # 5 个插值 + 1 个终点

    def test_endpoints_correct(self):
        """首尾点应与关键点一致。"""
        kps = np.array([[0, 0, 0], [2, 0, 1]], dtype=np.float32)
        result = graph_path_to_world([0, 1], kps, 3)
        assert abs(result[0][0]) < 0.01
        assert abs(result[-1][0] - 2.0) < 0.01
        assert abs(result[-1][2] - 1.0) < 0.01

    def test_empty_path(self):
        """空路径应返回空列表。"""
        kps = np.array([[0, 0, 0]], dtype=np.float32)
        result = graph_path_to_world([], kps, 5)
        assert len(result) == 0


class TestSmoothPath3D:
    """测试 3D 路径平滑。"""

    def test_short_path_unchanged(self):
        """2 点以下的路径不应改变。"""
        path = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        result = smooth_path_3d(path, 3, 0.5)
        assert len(result) == 2
        assert result == path

    def test_smoothing_reduces_roughness(self):
        """平滑应减少路径的锯齿。"""
        path = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.5, 0.0),
            (2.0, 0.0, 0.0),
        ]
        result = smooth_path_3d(path, 3, 0.5)
        assert len(result) == len(path)
        assert result[0] == path[0]
        assert result[-1] == path[-1]

    def test_height_weight_affects_z(self):
        """height_weight 应影响 Z 方向平滑程度。"""
        path = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
        ]
        result_full = smooth_path_3d(path, 3, 1.0)
        result_none = smooth_path_3d(path, 3, 0.0)
        assert abs(result_full[1][2] - 0.5) < abs(result_none[1][2] - 0.5)

    def test_straight_line_stays_straight(self):
        """直线平滑后应仍为直线。"""
        path = [(float(i), 0.0, 0.0) for i in range(10)]
        result = smooth_path_3d(path, 3, 0.5)
        for x, y, z in result:
            assert abs(y) < 1e-6
            assert abs(z) < 1e-6
