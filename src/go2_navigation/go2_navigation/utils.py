"""3D 导航工具函数：PCD 读取、体素化、图构建、路径搜索。"""

import heapq
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ── 数据结构 ─────────────────────────────────────────────────


@dataclass
class NavGraph:
    """3D 导航图：关键点 + 邻接关系。"""

    keypoints: np.ndarray  # (N, 3) float32, 关键点 XYZ 坐标
    heights: np.ndarray  # (N,) float32, 每个关键点的平均体素高度
    traversable: np.ndarray  # (N,) bool, 是否可通行
    tree: cKDTree = field(default=None, repr=False)  # cKDTree 空间索引
    adjacency: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    # adjacency[i] = [(j, cost), ...] 关键点 i 到邻居 j 的边和代价


# ── PCD 读取 ─────────────────────────────────────────────────


def read_pcd_binary(path: str) -> np.ndarray:
    """读取 binary 格式的 PCD 文件，返回 (N, 3) 的 xyz 坐标数组。

    仅支持 FIELDS 中包含 x y z 的 binary PCD。
    """
    header: dict[str, str] = {}
    data_offset = 0

    with open(path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('DATA'):
                data_offset = f.tell()
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = ' '.join(parts[1:])

    fields = header.get('FIELDS', 'x y z').split()
    sizes = [int(s) for s in header.get('SIZE', '4 4 4').split()]
    types = header.get('TYPE', 'F F F').split()
    counts = [int(c) for c in header.get('COUNT', '1 1 1').split()]
    points = int(header.get('POINTS', '0'))

    bytes_per_point = sum(s * c for s, c in zip(sizes, counts))

    offsets = []
    offset = 0
    for s, c in zip(sizes, counts):
        offsets.append(offset)
        offset += s * c

    xyz_indices = [fields.index(ax) for ax in ('x', 'y', 'z') if ax in fields]
    if len(xyz_indices) < 3:
        raise ValueError(f'PCD 文件缺少 x/y/z 字段: {fields}')

    xyz_offsets = [offsets[i] for i in xyz_indices]

    with open(path, 'rb') as f:
        f.seek(data_offset)
        raw = f.read(points * bytes_per_point)

    dtype_map = {'F': 'f4', 'U': 'u4', 'I': 'i4'}
    result = np.empty((points, 3), dtype=np.float32)

    for i in range(points):
        base = i * bytes_per_point
        for j, ax_offset in enumerate(xyz_offsets):
            result[i, j] = struct.unpack_from('<f', raw, base + ax_offset)[0]

    return result


# ── 体素化 ───────────────────────────────────────────────────


def auto_detect_ground_z(
    points: np.ndarray,
    band_width: float = 0.5,
    bin_size: float = 0.05,
) -> Tuple[float, float]:
    """自动检测地面高度：找 Z 分布最密集的区间。

    用直方图统计 Z 分布，找到点数最多的 bin，
    返回该 bin 中心 ± band_width/2 作为地面范围。

    Args:
        points: (N, 3) 完整点云。
        band_width: 地面带宽度 (m)，默认 0.5m。
        bin_size: 直方图 bin 大小 (m)，默认 0.05m。

    Returns:
        (ground_z_min, ground_z_max)
    """
    z = points[:, 2]
    # 用直方图找最密集的 Z 区间
    z_min, z_max = z.min(), z.max()
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    counts, edges = np.histogram(z, bins=bins)

    # 找点数最多的 bin
    peak_idx = np.argmax(counts)
    peak_z = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0

    ground_z_min = peak_z - band_width / 2.0
    ground_z_max = peak_z + band_width / 2.0

    return float(ground_z_min), float(ground_z_max)


def auto_calibrate_graph_params(
    keypoints: np.ndarray,
    neighbor_k: int = 3,
) -> Tuple[float, float]:
    """根据关键点分布自动校准图构建参数。

    计算每个关键点到最近邻居的距离，取 P95 值作为连接半径，
    确保几乎所有关键点都能连上至少一个邻居。

    Args:
        keypoints: (K, 3) 关键点坐标。
        neighbor_k: 用第 k 近邻居估算间距。

    Returns:
        (connection_radius, max_height_change)
    """
    if len(keypoints) < 2:
        return 0.5, 0.2

    tree = cKDTree(keypoints)
    k = min(neighbor_k + 1, len(keypoints))
    dists, _ = tree.query(keypoints, k=k)
    nn_dists = dists[:, 1]  # 到第 1 近邻居的距离（跳过自身）

    # 用 P95 确保覆盖稀疏区域，乘 1.2 留余量
    connection_radius = float(np.percentile(nn_dists, 95) * 1.2)
    connection_radius = max(0.5, connection_radius)
    max_height_change = max(0.2, connection_radius * 0.5)

    return float(connection_radius), float(max_height_change)


def voxelize_ground(
    points: np.ndarray,
    resolution: float = 0.2,
    min_points: int = 3,
    ground_z_max: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """将地面级点云体素化，返回有效体素的中心和高度。

    Args:
        points: (M, 3) 原始点云（map 坐标系）。
        resolution: 体素边长（米）。
        min_points: 体素内最少点数才算有效。
        ground_z_max: 地面点的最大 Z 值，超过此高度的点不参与体素化。

    Returns:
        voxel_centers: (N, 3) 有效体素的中心坐标。
        voxel_heights: (N,) 每个体素的平均 Z 高度。
    """
    # 过滤地面带
    ground_mask = points[:, 2] <= ground_z_max
    ground_pts = points[ground_mask]

    if len(ground_pts) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    # 计算整数体素坐标
    voxel_coords = np.floor(ground_pts / resolution).astype(np.int32)

    # 按体素分组
    unique_coords, inverse, counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True
    )

    # 过滤最少点数
    valid_mask = counts >= min_points
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    # 计算每个有效体素的平均位置
    voxel_centers = np.zeros((len(valid_indices), 3), dtype=np.float32)
    voxel_heights = np.zeros(len(valid_indices), dtype=np.float32)

    for new_idx, old_idx in enumerate(valid_indices):
        mask = inverse == old_idx
        pts_in_voxel = ground_pts[mask]
        voxel_centers[new_idx] = pts_in_voxel.mean(axis=0)
        voxel_heights[new_idx] = pts_in_voxel[:, 2].mean()

    return voxel_centers, voxel_heights


def extract_keypoints(
    voxel_centers: np.ndarray,
    voxel_heights: np.ndarray,
    downsample_res: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """降采样体素中心，提取导航关键点。

    Args:
        voxel_centers: (N, 3) 体素中心。
        voxel_heights: (N,) 体素高度。
        downsample_res: 关键点间距（米）。

    Returns:
        keypoint_positions: (K, 3) 关键点坐标。
        keypoint_heights: (K,) 关键点高度。
    """
    if len(voxel_centers) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    # 用更大体素做降采样
    ds_coords = np.floor(voxel_centers / downsample_res).astype(np.int32)
    unique_ds, inverse_ds, counts_ds = np.unique(
        ds_coords, axis=0, return_inverse=True, return_counts=True
    )

    keypoint_positions = np.zeros((len(unique_ds), 3), dtype=np.float32)
    keypoint_heights = np.zeros(len(unique_ds), dtype=np.float32)

    for i in range(len(unique_ds)):
        mask = inverse_ds == i
        keypoint_positions[i] = voxel_centers[mask].mean(axis=0)
        keypoint_heights[i] = voxel_heights[mask].mean()

    return keypoint_positions, keypoint_heights


# ── 图构建 ───────────────────────────────────────────────────


def build_nav_graph(
    keypoints: np.ndarray,
    heights: np.ndarray,
    connection_radius: float = 0.5,
    max_height_change: float = 0.2,
    max_slope_angle: float = 45.0,
    keep_largest_component_only: bool = True,
    min_component_size: int = 1,
) -> NavGraph:
    """构建 KD-tree 和邻接图。

    对每个关键点，查询 connection_radius 内的邻居，过滤高差和坡度，
    生成带权邻接图。

    Args:
        keypoints: (K, 3) 关键点坐标。
        heights: (K,) 关键点高度。
        connection_radius: 最大边长（米）。
        max_height_change: 最大允许高差（米）。
        max_slope_angle: 最大允许坡度（度）。

    Returns:
        NavGraph 对象。
    """
    if len(keypoints) == 0:
        graph = NavGraph(
            keypoints=keypoints,
            heights=heights,
            traversable=np.empty(0, dtype=bool),
        )
        return graph

    tree = cKDTree(keypoints)
    adjacency: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(len(keypoints)):
        neighbor_indices = tree.query_ball_point(keypoints[i], connection_radius)
        edges: List[Tuple[int, float]] = []

        for j in neighbor_indices:
            if i == j:
                continue

            dx = keypoints[j, 0] - keypoints[i, 0]
            dy = keypoints[j, 1] - keypoints[i, 1]
            dz = keypoints[j, 2] - keypoints[i, 2]

            # 高差过滤
            if abs(dz) > max_height_change:
                continue

            # 坡度过滤
            horiz_dist = np.sqrt(dx * dx + dy * dy)
            if horiz_dist < 1e-6:
                continue
            slope_deg = np.degrees(np.arctan2(abs(dz), horiz_dist))
            if slope_deg > max_slope_angle:
                continue

            # 代价 = 欧氏距离 × 高度惩罚系数
            euclidean = np.sqrt(dx * dx + dy * dy + dz * dz)
            height_penalty = 1.0 + 4.0 * (abs(dz) / max(max_height_change, 1e-6))
            cost = euclidean * height_penalty

            edges.append((j, cost))

        adjacency[i] = edges

    traversable = np.ones(len(keypoints), dtype=bool)

    # 找连通分量，剔除孤立点和小分量
    visited = set()
    components: List[List[int]] = []
    for i in range(len(keypoints)):
        if i in visited:
            continue
        component: List[int] = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor, _ in adjacency.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)

    if keep_largest_component_only:
        keep_components = [max(components, key=len)] if components else []
    else:
        min_size = max(1, int(min_component_size))
        keep_components = [component for component in components if len(component) >= min_size]
        if not keep_components and components:
            keep_components = [max(components, key=len)]

    keep_set = {idx for component in keep_components for idx in component}
    keep_mask = np.array([i in keep_set for i in range(len(keypoints))])

    if not np.all(keep_mask):
        old_to_new = -np.ones(len(keypoints), dtype=int)
        new_idx = 0
        for i in range(len(keypoints)):
            if keep_mask[i]:
                old_to_new[i] = new_idx
                new_idx += 1

        new_adjacency: Dict[int, List[Tuple[int, float]]] = {}
        for old_i in range(len(keypoints)):
            new_i = old_to_new[old_i]
            if new_i < 0:
                continue
            edges = []
            for old_j, cost in adjacency.get(old_i, []):
                new_j = old_to_new[old_j]
                if new_j >= 0:
                    edges.append((new_j, cost))
            new_adjacency[new_i] = edges

        keypoints = keypoints[keep_mask]
        heights = heights[keep_mask]
        traversable = np.ones(len(keypoints), dtype=bool)
        tree = cKDTree(keypoints)
        adjacency = new_adjacency

    return NavGraph(
        keypoints=keypoints,
        heights=heights,
        traversable=traversable,
        tree=tree,
        adjacency=adjacency,
    )


def find_nearest_keypoint(tree: cKDTree, position: Tuple[float, float, float]) -> int:
    """查找离给定 3D 位置最近的关键点索引。"""
    _, idx = tree.query(position)
    return int(idx)


# ── 路径搜索 ─────────────────────────────────────────────────


def dijkstra_search(
    adjacency: Dict[int, List[Tuple[int, float]]],
    start_idx: int,
    goal_idx: int,
) -> Optional[List[int]]:
    """在 3D 关键点图上执行 Dijkstra 搜索。

    Args:
        adjacency: 邻接图 {节点: [(邻居, 代价), ...]}。
        start_idx: 起点关键点索引。
        goal_idx: 终点关键点索引。

    Returns:
        关键点索引路径 [start, ..., goal]，或 None（不可达）。
    """
    g_score: Dict[int, float] = {start_idx: 0.0}
    came_from: Dict[int, int] = {}
    open_set: List[Tuple[float, int]] = [(0.0, start_idx)]

    while open_set:
        current_g, current = heapq.heappop(open_set)

        if current == goal_idx:
            path = []
            node = goal_idx
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start_idx)
            path.reverse()
            return path

        if current_g > g_score.get(current, float('inf')):
            continue

        for neighbor, edge_cost in adjacency.get(current, []):
            new_g = current_g + edge_cost
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_g, neighbor))

    return None


# ── 路径处理 ─────────────────────────────────────────────────


def graph_path_to_world(
    path_indices: List[int],
    keypoints: np.ndarray,
    num_interp: int = 5,
) -> List[Tuple[float, float, float]]:
    """将图路径索引转换为插值后的 3D 世界坐标。

    在关键点之间插入中间点使路径更平滑。

    Args:
        path_indices: 关键点索引列表。
        keypoints: (K, 3) 关键点坐标。
        num_interp: 每段边的插值点数。

    Returns:
        [(x, y, z), ...] 世界坐标路径。
    """
    if len(path_indices) < 2:
        if len(path_indices) == 1:
            p = keypoints[path_indices[0]]
            return [(float(p[0]), float(p[1]), float(p[2]))]
        return []

    world_path: List[Tuple[float, float, float]] = []

    for seg in range(len(path_indices) - 1):
        p0 = keypoints[path_indices[seg]]
        p1 = keypoints[path_indices[seg + 1]]

        for t_idx in range(num_interp):
            t = t_idx / num_interp
            x = p0[0] + t * (p1[0] - p0[0])
            y = p0[1] + t * (p1[1] - p0[1])
            z = p0[2] + t * (p1[2] - p0[2])
            world_path.append((float(x), float(y), float(z)))

    # 添加最后一个点
    p_last = keypoints[path_indices[-1]]
    world_path.append((float(p_last[0]), float(p_last[1]), float(p_last[2])))

    return world_path


def smooth_path_3d(
    path: List[Tuple[float, float, float]],
    iterations: int = 3,
    height_weight: float = 0.5,
) -> List[Tuple[float, float, float]]:
    """3D 路径移动平均平滑。

    XY 方向全权重平滑，Z 方向降低权重以保留垂直结构。

    Args:
        path: [(x, y, z), ...] 输入路径。
        iterations: 平滑迭代次数。
        height_weight: Z 方向平滑权重（0-1）。

    Returns:
        平滑后的路径。
    """
    if len(path) <= 2:
        return list(path)

    pts = [list(p) for p in path]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            sx = 0.25 * pts[i - 1][0] + 0.5 * pts[i][0] + 0.25 * pts[i + 1][0]
            sy = 0.25 * pts[i - 1][1] + 0.5 * pts[i][1] + 0.25 * pts[i + 1][1]
            # Z 用降低权重的平滑，保留坡道等垂直结构
            sz_interp = (
                0.25 * pts[i - 1][2] + 0.5 * pts[i][2] + 0.25 * pts[i + 1][2]
            )
            sz = pts[i][2] + height_weight * (sz_interp - pts[i][2])
            new_pts.append([sx, sy, sz])
        new_pts.append(pts[-1])
        pts = new_pts

    return [(p[0], p[1], p[2]) for p in pts]
