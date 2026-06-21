"""3D 地图服务器：加载 PCD 点云，构建导航关键点图并发布。

用法:
    ros2 run go2_navigation map_server --ros-args -p pcd_path:=/path/to/map.pcd
    或通过 launch 文件启动（自动加载参数）。
"""

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
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
        self.declare_parameter('auto_calibrate', True)
        self.declare_parameter('nav_graph_topic', '/nav_graph_cloud')
        self.declare_parameter('map_frame', 'map')

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
        self.auto_calibrate = self.get_parameter('auto_calibrate').value
        nav_graph_topic = self.get_parameter('nav_graph_topic').value
        self.map_frame = self.get_parameter('map_frame').value

        if not pcd_path:
            pcd_path = '/home/w/C206Go2/src/FAST_LIO_ROS2/PCD/test.pcd'

        self.get_logger().info(f'加载 PCD: {pcd_path}')

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

        # 定时发布导航图和原始地图
        self.create_timer(1.0, self._publish_nav_graph)
        self.create_timer(1.0, self._publish_map_cloud)

        # ── 重载服务 ──
        self.create_service(Trigger, '/map_server/reload', self._reload_cb)

        self.get_logger().info('3D 地图服务器已启动')

    # ────────────────────────────────────────────────────────────
    # 3D 导航图构建
    # ────────────────────────────────────────────────────────────

    def _build_nav_graph(self, pcd_path: str) -> NavGraph:
        """加载 PCD → 体素化 → 提取关键点 → 构建导航图。"""
        points = read_pcd_binary(pcd_path)
        self.get_logger().info(f'读取 {len(points)} 个点')

        # 自动检测地面高度
        if self.auto_ground:
            self.ground_z_min, self.ground_z_max = auto_detect_ground_z(
                points, self.ground_band_width
            )
            self.get_logger().info(
                f'自动检测地面: Z=[{self.ground_z_min:.2f}, {self.ground_z_max:.2f}]m'
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
