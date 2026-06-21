"""3D 地图服务器：加载 PCD 点云，构建导航关键点图并发布。

用法:
    ros2 run go2_navigation map_server --ros-args -p pcd_path:=/path/to/map.pcd
    或通过 launch 文件启动（自动加载参数）。
"""

import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from scipy.spatial import cKDTree
from scipy import ndimage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger
from std_msgs.msg import Header

from .utils import (
    NavGraph,
    read_pcd_binary,
    voxelize_ground,
    extract_keypoints,
    build_nav_graph,
    auto_detect_ground_z,
    auto_calibrate_graph_params,
)


class MapServer(Node):
    def __init__(self) -> None:
        super().__init__('map_server')

        # ── 参数声明 ──
        self.declare_parameter('pcd_path', '')
        self.declare_parameter('voxel_resolution', 0.2)
        self.declare_parameter('min_points_per_voxel', 3)
        self.declare_parameter('keypoint_resolution', 0.2)
        self.declare_parameter('connection_radius', 0.5)
        self.declare_parameter('max_height_change', 0.2)
        self.declare_parameter('max_slope_angle', 45.0)
        self.declare_parameter('ground_z_min', 0.0)
        self.declare_parameter('ground_z_max', 0.5)
        self.declare_parameter('auto_ground_detect', True)
        self.declare_parameter('ground_band_width', 0.5)
        self.declare_parameter('floor_detection_mode', 'odom')
        self.declare_parameter('odom_topic', '/Odometry_base_link')
        self.declare_parameter('robot_base_height', 0.30)
        self.declare_parameter('ground_band_below', 0.20)
        self.declare_parameter('ground_band_above', 0.25)
        self.declare_parameter('odom_wait_timeout', 3.0)
        self.declare_parameter('auto_calibrate', True)
        self.declare_parameter('nav_graph_topic', '/nav_graph_cloud')
        self.declare_parameter('map_frame', 'camera_init')
        self.declare_parameter('nav_graph_source', 'ground_points')
        self.declare_parameter('free_space_resolution', 0.4)
        self.declare_parameter('robot_clearance', 0.38)
        self.declare_parameter('obstacle_min_height', 0.12)
        self.declare_parameter('obstacle_max_height', 1.0)
        self.declare_parameter('free_space_padding', 0.2)
        self.declare_parameter('free_space_bounds_percentile', 1.0)
        self.declare_parameter('max_unknown_distance', 1.6)
        self.declare_parameter('ground_support_radius', 1.0)
        self.declare_parameter('min_component_cells', 80)

        pcd_path = self.get_parameter('pcd_path').get_parameter_value().string_value
        self.voxel_resolution = self.get_parameter('voxel_resolution').value
        self.min_points_per_voxel = self.get_parameter('min_points_per_voxel').value
        self.keypoint_resolution = self.get_parameter('keypoint_resolution').value
        self.connection_radius = self.get_parameter('connection_radius').value
        self.max_height_change = self.get_parameter('max_height_change').value
        self.max_slope_angle = self.get_parameter('max_slope_angle').value
        self.ground_z_min = self.get_parameter('ground_z_min').value
        self.ground_z_max = self.get_parameter('ground_z_max').value
        self.auto_ground = self.get_parameter('auto_ground_detect').value
        self.ground_band_width = self.get_parameter('ground_band_width').value
        self.floor_detection_mode = self.get_parameter('floor_detection_mode').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.robot_base_height = self.get_parameter('robot_base_height').value
        self.ground_band_below = self.get_parameter('ground_band_below').value
        self.ground_band_above = self.get_parameter('ground_band_above').value
        self.odom_wait_timeout = self.get_parameter('odom_wait_timeout').value
        self.auto_calibrate = self.get_parameter('auto_calibrate').value
        nav_graph_topic = self.get_parameter('nav_graph_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.nav_graph_source = self.get_parameter('nav_graph_source').value
        self.free_space_resolution = self.get_parameter('free_space_resolution').value
        self.robot_clearance = self.get_parameter('robot_clearance').value
        self.obstacle_min_height = self.get_parameter('obstacle_min_height').value
        self.obstacle_max_height = self.get_parameter('obstacle_max_height').value
        self.free_space_padding = self.get_parameter('free_space_padding').value
        self.free_space_bounds_percentile = self.get_parameter(
            'free_space_bounds_percentile'
        ).value
        self.max_unknown_distance = self.get_parameter('max_unknown_distance').value
        self.ground_support_radius = self.get_parameter('ground_support_radius').value
        self.min_component_cells = self.get_parameter('min_component_cells').value
        self._floor_from_odom = False
        self._last_base_z = None

        if not pcd_path:
            pcd_path = '/home/w/C206Go2/src/FAST_LIO-ROS2/PCD/scan.pcd'

        self.get_logger().info(f'加载 PCD: {pcd_path}')
        self._set_ground_band_from_odom_if_available()

        # ── 构建 3D 导航图 ──
        self.nav_graph: NavGraph = self._build_nav_graph(pcd_path)
        self.get_logger().info(
            f'导航图: {len(self.nav_graph.keypoints)} 个关键点, '
            f'{sum(len(v) for v in self.nav_graph.adjacency.values())} 条边'
        )

        # ── 发布者 ──
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.nav_graph_pub = self.create_publisher(
            PointCloud2, nav_graph_topic, latched_qos
        )

        self.map_cloud_pub = self.create_publisher(
            PointCloud2, '/map_cloud', latched_qos
        )

        # 保存地图数据，定时发布
        self._map_cloud_points = read_pcd_binary(pcd_path)[::4]

        # Static map topics are latched; publish once to avoid RViz flicker.
        self._initial_publish_done = False
        self._initial_publish_timer = self.create_timer(0.5, self._publish_once)

        # ── 重载服务 ──
        self.create_service(Trigger, '/map_server/reload', self._reload_cb)

        self.get_logger().info('3D 地图服务器已启动')

    def _publish_once(self) -> None:
        if self._initial_publish_done:
            return
        self._publish_nav_graph()
        self._publish_map_cloud()
        self._initial_publish_done = True
        self.destroy_timer(self._initial_publish_timer)

    # ────────────────────────────────────────────────────────────
    # 3D 导航图构建
    # ────────────────────────────────────────────────────────────

    def _build_nav_graph(self, pcd_path: str) -> NavGraph:
        """加载 PCD → 体素化 → 提取关键点 → 构建导航图。"""
        points = read_pcd_binary(pcd_path)
        self.get_logger().info(f'读取 {len(points)} 个点')

        # 自动检测地面高度
        if self.auto_ground and not self._floor_from_odom:
            self.ground_z_min, self.ground_z_max = auto_detect_ground_z(
                points, self.ground_band_width
            )
            self.get_logger().info(
                f'自动检测地面: Z=[{self.ground_z_min:.2f}, {self.ground_z_max:.2f}]m'
            )
        elif self._floor_from_odom:
            self.get_logger().info(
                f'按当前楼层提取地面: Z=[{self.ground_z_min:.2f}, {self.ground_z_max:.2f}]m'
            )
        else:
            self.get_logger().info(
                f'使用手动地面高度: Z=[{self.ground_z_min:.2f}, {self.ground_z_max:.2f}]m'
            )

        if self.nav_graph_source == 'ground_grid':
            return self._build_ground_grid_nav_graph(points)
        if self.nav_graph_source == 'free_space':
            return self._build_free_space_nav_graph(points)
        if self.nav_graph_source != 'ground_points':
            self.get_logger().warn(
                f'未知 nav_graph_source={self.nav_graph_source}，回退 ground_points'
            )

        # 过滤地面点
        ground_mask = (points[:, 2] >= self.ground_z_min) & (
            points[:, 2] <= self.ground_z_max
        )
        ground_pts = points[ground_mask]
        self.get_logger().info(f'地面点: {len(ground_pts)}')

        if len(ground_pts) == 0:
            self.get_logger().error('地面点为空！请检查 ground_z_min/max 参数')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        # 体素化
        voxel_centers, voxel_heights = voxelize_ground(
            ground_pts, self.voxel_resolution, self.min_points_per_voxel, self.ground_z_max
        )
        self.get_logger().info(f'有效体素: {len(voxel_centers)}')

        # 提取关键点
        kp_positions, kp_heights = extract_keypoints(
            voxel_centers, voxel_heights, self.keypoint_resolution
        )
        self.get_logger().info(f'导航关键点: {len(kp_positions)}')

        # 自动校准图参数
        if self.auto_calibrate:
            self.connection_radius, self.max_height_change = auto_calibrate_graph_params(
                kp_positions
            )
            self.get_logger().info(
                f'自动校准: connection_radius={self.connection_radius:.2f}m, '
                f'max_height_change={self.max_height_change:.2f}m'
            )

        # 构建图
        graph = build_nav_graph(
            kp_positions,
            kp_heights,
            self.connection_radius,
            self.max_height_change,
            self.max_slope_angle,
        )

        return graph

    def _build_ground_grid_nav_graph(self, points: np.ndarray) -> NavGraph:
        """用真实地面点生成可通行高程栅格。

        这个模式仍然使用 3D 点云：先按 Z 高度提取当前楼层地面点，再在 XY
        平面栅格化。它只对地面小缺口做闭运算补洞，不会把未知大空洞当成
        可通行区域，最后再用障碍物点膨胀出安全距离。
        """
        ground_mask = (points[:, 2] >= self.ground_z_min) & (
            points[:, 2] <= self.ground_z_max
        )
        ground_pts = points[ground_mask]
        if len(ground_pts) == 0:
            self.get_logger().error('地面点为空，无法构建地面栅格导航图')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        floor_z = float(np.median(ground_pts[:, 2]))
        res = float(self.free_space_resolution)
        pct = float(self.free_space_bounds_percentile)
        pct = max(0.0, min(10.0, pct))
        ground_xy = ground_pts[:, :2]
        ground_xy = ground_xy[np.isfinite(ground_xy).all(axis=1)]

        x_min, y_min = np.percentile(ground_xy, pct, axis=0) - self.free_space_padding
        x_max, y_max = np.percentile(ground_xy, 100.0 - pct, axis=0) + self.free_space_padding
        nx = int(np.floor((x_max - x_min) / res)) + 1
        ny = int(np.floor((y_max - y_min) / res)) + 1
        if nx <= 0 or ny <= 0:
            self.get_logger().error('地面栅格边界为空，无法构建导航图')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        gx = np.floor((ground_xy[:, 0] - x_min) / res).astype(np.int32)
        gy = np.floor((ground_xy[:, 1] - y_min) / res).astype(np.int32)
        valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        gx = gx[valid]
        gy = gy[valid]

        ground_counts = np.zeros((ny, nx), dtype=np.int32)
        np.add.at(ground_counts, (gy, gx), 1)
        ground_grid = ground_counts >= int(max(1, self.min_points_per_voxel))

        close_iters = max(1, int(round(float(self.ground_support_radius) / max(res, 1e-6) / 2.0)))
        close_iters = min(close_iters, 3)
        structure = np.ones((3, 3), dtype=bool)
        supported_grid = ndimage.binary_closing(
            ground_grid, structure=structure, iterations=close_iters
        )

        obs_z_min = floor_z + float(self.obstacle_min_height)
        obs_z_max = floor_z + float(self.obstacle_max_height)
        obstacle_mask = (points[:, 2] >= obs_z_min) & (points[:, 2] <= obs_z_max)
        obstacle_xy = points[obstacle_mask, :2]
        obstacle_xy = obstacle_xy[np.isfinite(obstacle_xy).all(axis=1)]

        obstacle_grid = np.zeros((ny, nx), dtype=bool)
        if len(obstacle_xy) > 0:
            ox = np.floor((obstacle_xy[:, 0] - x_min) / res).astype(np.int32)
            oy = np.floor((obstacle_xy[:, 1] - y_min) / res).astype(np.int32)
            valid_obs = (ox >= 0) & (ox < nx) & (oy >= 0) & (oy < ny)
            obstacle_grid[oy[valid_obs], ox[valid_obs]] = True

        inflate_iters = max(0, int(np.ceil(float(self.robot_clearance) / max(res, 1e-6))))
        if inflate_iters > 0:
            inflated_obstacles = ndimage.binary_dilation(
                obstacle_grid, structure=structure, iterations=inflate_iters
            )
        else:
            inflated_obstacles = obstacle_grid

        free_grid = supported_grid & ~inflated_obstacles
        labeled, n_labels = ndimage.label(free_grid, structure=structure)
        if n_labels > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            min_cells = int(max(1, self.min_component_cells))
            keep_labels = np.where(sizes >= min_cells)[0]
            if len(keep_labels) == 0:
                keep_labels = np.array([int(np.argmax(sizes))])
            free_grid = np.isin(labeled, keep_labels)

        ys, xs = np.where(free_grid)
        keypoints = np.zeros((len(xs), 3), dtype=np.float32)
        keypoints[:, 0] = x_min + (xs.astype(np.float32) + 0.5) * res
        keypoints[:, 1] = y_min + (ys.astype(np.float32) + 0.5) * res
        keypoints[:, 2] = floor_z
        heights = np.full(len(keypoints), floor_z, dtype=np.float32)

        self.get_logger().info(
            f'地面栅格模式: floor_z={floor_z:.2f}m, 地面点={len(ground_pts)}, '
            f'原始地面格={int(ground_grid.sum())}, 补洞后={int(supported_grid.sum())}, '
            f'障碍格={int(obstacle_grid.sum())}, 可通行格={len(keypoints)}, '
            f'补洞迭代={close_iters}, 膨胀迭代={inflate_iters}'
        )

        if len(keypoints) == 0:
            self.get_logger().error('地面栅格可通行点为空，请检查地面高度或减小 robot_clearance')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        connection_radius = max(self.connection_radius, res * 1.5)
        max_height_change = max(self.max_height_change, 0.2)
        return build_nav_graph(
            keypoints,
            heights,
            connection_radius,
            max_height_change,
            self.max_slope_angle,
        )

    def _build_free_space_nav_graph(self, points: np.ndarray) -> NavGraph:
        """用障碍物投影生成 2D 自由空间导航图。

        MID360 在走廊地面上的回波常常很稀疏，直接把地面点当可走点会让
        /nav_graph_cloud 变成散点。这里反过来处理：把墙、柱子、杂物等
        高于地面的点投影为障碍物，膨胀出安全距离，再在剩余自由空间采样。
        """
        ground_mask = (points[:, 2] >= self.ground_z_min) & (
            points[:, 2] <= self.ground_z_max
        )
        ground_pts = points[ground_mask]
        if len(ground_pts) > 0:
            floor_z = float(np.median(ground_pts[:, 2]))
        else:
            floor_z = float((self.ground_z_min + self.ground_z_max) * 0.5)

        obs_z_min = floor_z + float(self.obstacle_min_height)
        obs_z_max = floor_z + float(self.obstacle_max_height)
        obstacle_mask = (points[:, 2] >= obs_z_min) & (points[:, 2] <= obs_z_max)
        obstacle_pts = points[obstacle_mask]

        support_mask = ground_mask | obstacle_mask
        support_pts = points[support_mask]
        if len(support_pts) == 0:
            support_pts = points

        self.get_logger().info(
            f'自由空间模式: floor_z={floor_z:.2f}m, '
            f'障碍物Z=[{obs_z_min:.2f}, {obs_z_max:.2f}]m, '
            f'障碍点={len(obstacle_pts)}, 支撑点={len(support_pts)}'
        )

        bounds_pts = ground_pts if len(ground_pts) > 0 else support_pts
        xy = bounds_pts[:, :2]
        finite_mask = np.isfinite(xy).all(axis=1)
        xy = xy[finite_mask]
        if len(xy) == 0:
            self.get_logger().error('自由空间支撑点为空，无法构建导航图')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        pct = float(self.free_space_bounds_percentile)
        pct = max(0.0, min(10.0, pct))
        x_min, y_min = np.percentile(xy, pct, axis=0) - self.free_space_padding
        x_max, y_max = np.percentile(xy, 100.0 - pct, axis=0) + self.free_space_padding

        res = float(self.free_space_resolution)
        xs = np.arange(x_min, x_max + res, res, dtype=np.float32)
        ys = np.arange(y_min, y_max + res, res, dtype=np.float32)
        if len(xs) == 0 or len(ys) == 0:
            self.get_logger().error('自由空间边界为空，无法构建导航图')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
        candidates_xy = np.column_stack(
            [grid_x.ravel(), grid_y.ravel()]
        ).astype(np.float32)

        free_mask = np.ones(len(candidates_xy), dtype=bool)
        if len(obstacle_pts) > 0:
            obstacle_xy = obstacle_pts[:, :2]
            obstacle_xy = obstacle_xy[np.isfinite(obstacle_xy).all(axis=1)]
            if len(obstacle_xy) > 0:
                obstacle_tree = cKDTree(obstacle_xy)
                obs_dist, _ = obstacle_tree.query(candidates_xy, k=1)
                free_mask &= obs_dist >= float(self.robot_clearance)

        if self.ground_support_radius > 0.0 and len(ground_pts) > 0:
            ground_xy = ground_pts[:, :2]
            ground_xy = ground_xy[np.isfinite(ground_xy).all(axis=1)]
            if len(ground_xy) > 0:
                ground_tree = cKDTree(ground_xy)
                ground_dist, _ = ground_tree.query(candidates_xy, k=1)
                free_mask &= ground_dist <= float(self.ground_support_radius)
        elif self.max_unknown_distance > 0.0:
            self.get_logger().warn(
                '地面支撑点为空，临时回退到点云支撑距离；导航图可能包含未知区域'
            )
            support_tree = cKDTree(xy)
            support_dist, _ = support_tree.query(candidates_xy, k=1)
            free_mask &= support_dist <= float(self.max_unknown_distance)

        free_xy = candidates_xy[free_mask]
        keypoints = np.zeros((len(free_xy), 3), dtype=np.float32)
        keypoints[:, :2] = free_xy
        keypoints[:, 2] = floor_z
        heights = np.full(len(keypoints), floor_z, dtype=np.float32)

        self.get_logger().info(
            f'自由空间采样: 候选={len(candidates_xy)}, 可通行={len(keypoints)}, '
            f'分辨率={res:.2f}m, 安全距离={self.robot_clearance:.2f}m, '
            f'地面支撑半径={self.ground_support_radius:.2f}m'
        )

        if len(keypoints) == 0:
            self.get_logger().error('自由空间点为空，请减小 robot_clearance 或检查地面高度')
            return NavGraph(
                keypoints=np.empty((0, 3), dtype=np.float32),
                heights=np.empty(0, dtype=np.float32),
                traversable=np.empty(0, dtype=bool),
            )

        connection_radius = self.connection_radius
        max_height_change = self.max_height_change
        if self.auto_calibrate:
            connection_radius = max(connection_radius, res * 1.45)
            max_height_change = max(max_height_change, 0.2)
            self.get_logger().info(
                f'自由空间图参数: connection_radius={connection_radius:.2f}m, '
                f'max_height_change={max_height_change:.2f}m'
            )

        return build_nav_graph(
            keypoints,
            heights,
            connection_radius,
            max_height_change,
            self.max_slope_angle,
        )

    # ────────────────────────────────────────────────────────────
    # 当前楼层地面高度估计
    # ────────────────────────────────────────────────────────────

    def _set_ground_band_from_odom_if_available(self) -> None:
        """用 base_link 高度估计当前楼层地面高度。

        Go2 的 base_link 在身体附近，站立时比地面高约 robot_base_height。
        因此当前楼层地面高度约为 base_link_z - robot_base_height。
        """
        if self.floor_detection_mode == 'manual':
            return
        if self.floor_detection_mode == 'auto':
            return
        if self.floor_detection_mode != 'odom':
            self.get_logger().warn(
                f'未知 floor_detection_mode={self.floor_detection_mode}，回退自动检测'
            )
            return

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5,
        )
        sub = self.create_subscription(
            Odometry, self.odom_topic, self._on_floor_odom, qos
        )

        deadline = time.monotonic() + float(self.odom_wait_timeout)
        while self._last_base_z is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.destroy_subscription(sub)

        if self._last_base_z is None:
            self.get_logger().warn(
                f'{self.odom_wait_timeout:.1f}s 内未收到 {self.odom_topic}，'
                '回退到点云 Z 分布自动检测地面'
            )
            return

        floor_z = float(self._last_base_z) - float(self.robot_base_height)
        self.ground_z_min = floor_z - float(self.ground_band_below)
        self.ground_z_max = floor_z + float(self.ground_band_above)
        self._floor_from_odom = True
        self.get_logger().info(
            f'收到 base_link_z={self._last_base_z:.2f}m，'
            f'按 base_link 离地 {self.robot_base_height:.2f}m 估计当前楼层地面 '
            f'floor_z={floor_z:.2f}m'
        )

    def _on_floor_odom(self, msg: Odometry) -> None:
        self._last_base_z = msg.pose.pose.position.z

    # ────────────────────────────────────────────────────────────
    # 发布原始地图
    # ────────────────────────────────────────────────────────────

    def _publish_map_cloud(self) -> None:
        """发布原始 PCD 点云到 /map_cloud，供 RViz 显示完整地图。"""
        points = self._map_cloud_points
        n_points = len(points)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]

        msg = PointCloud2(
            header=header,
            height=1,
            width=n_points,
            fields=fields,
            is_bigendian=False,
            point_step=12,  # 3 个 float32
            row_step=12 * n_points,
            is_dense=True,
            data=data.tobytes(),
        )

        self.map_cloud_pub.publish(msg)
        self.get_logger().info(f'原始地图已发布: {n_points} 点 (/map_cloud)')

    # ────────────────────────────────────────────────────────────
    # 发布导航图
    # ────────────────────────────────────────────────────────────

    def _publish_nav_graph(self) -> None:
        """发布导航关键点云到 /nav_graph_cloud。"""
        kps = self.nav_graph.keypoints
        if len(kps) == 0:
            return

        # 构建 PointCloud2 消息
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        # 字段: x, y, z, intensity（用于标志位）
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]

        # 打包数据: x, y, z, intensity
        n_points = len(kps)
        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        data['x'] = kps[:, 0]
        data['y'] = kps[:, 1]
        data['z'] = kps[:, 2]
        # intensity 编码可通行标志: 1.0=可通行, 0.0=不可通行
        data['intensity'] = self.nav_graph.traversable.astype(np.float32)

        msg = PointCloud2(
            header=header,
            height=1,
            width=n_points,
            fields=fields,
            is_bigendian=False,
            point_step=16,  # 4 个 float32
            row_step=16 * n_points,
            is_dense=True,
            data=data.tobytes(),
        )

        self.nav_graph_pub.publish(msg)

    # ────────────────────────────────────────────────────────────
    # 重载服务
    # ────────────────────────────────────────────────────────────

    def _reload_cb(self, request, response):
        pcd_path = (
            self.get_parameter('pcd_path').get_parameter_value().string_value
        )
        try:
            self.nav_graph = self._build_nav_graph(pcd_path)
            response.success = True
            response.message = '地图已重载'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MapServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
