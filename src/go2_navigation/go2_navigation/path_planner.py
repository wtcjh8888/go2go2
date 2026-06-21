"""3D 路径规划器：在导航关键点图上执行 Dijkstra 搜索。

订阅 /nav_graph_cloud 和补偿后的 Odometry，接收 /goal_pose 目标点，
发布 /planned_path 规划结果。
"""

from typing import List, Optional, Tuple

import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from .utils import (
    NavGraph,
    build_nav_graph,
    find_nearest_keypoint,
    dijkstra_search,
    graph_path_to_world,
    smooth_path_3d,
    auto_calibrate_graph_params,
)


class PathPlanner(Node):
    def __init__(self) -> None:
        super().__init__('path_planner')

        self.declare_parameter('planner_frequency', 1.0)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('smooth_iterations', 3)
        self.declare_parameter('graph_interp_points', 5)
        self.declare_parameter('connection_radius', 0.5)
        self.declare_parameter('max_height_change', 0.2)
        self.declare_parameter('max_slope_angle', 45.0)
        self.declare_parameter('auto_calibrate', True)
        self.declare_parameter('odom_topic', '/Odometry_base_link')

        self.planner_freq = self.get_parameter('planner_frequency').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.smooth_iters = self.get_parameter('smooth_iterations').value
        self.interp_points = self.get_parameter('graph_interp_points').value
        self.connection_radius = self.get_parameter('connection_radius').value
        self.max_height_change = self.get_parameter('max_height_change').value
        self.max_slope_angle = self.get_parameter('max_slope_angle').value
        self.auto_calibrate = self.get_parameter('auto_calibrate').value
        self.odom_topic = self.get_parameter('odom_topic').value

        # 状态
        self.nav_graph: Optional[NavGraph] = None
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_z: float = 0.0
        self.goal: Optional[Tuple[float, float, float]] = None

        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        self.create_subscription(
            PointCloud2, '/nav_graph_cloud', self._on_nav_graph, map_qos
        )
        self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, odom_qos
        )
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.create_timer(1.0 / self.planner_freq, self._plan_tick)

        self.get_logger().info(
            f'3D 路径规划器已启动，订阅里程计: {self.odom_topic}'
        )

    def _on_nav_graph(self, msg: PointCloud2) -> None:
        """接收导航关键点云，重建图结构。"""
        n_points = msg.width
        if n_points == 0:
            return

        # 解析 PointCloud2
        kps = np.zeros((n_points, 3), dtype=np.float32)
        traversable = np.zeros(n_points, dtype=bool)

        point_step = msg.point_step
        data = msg.data

        for i in range(n_points):
            base = i * point_step
            x = struct.unpack_from('<f', data, base)[0]
            y = struct.unpack_from('<f', data, base + 4)[0]
            z = struct.unpack_from('<f', data, base + 8)[0]
            intensity = struct.unpack_from('<f', data, base + 12)[0]
            kps[i] = [x, y, z]
            traversable[i] = intensity > 0.5

        # 自动校准图参数
        if self.auto_calibrate:
            self.connection_radius, self.max_height_change = auto_calibrate_graph_params(
                kps
            )

        # 构建图
        self.nav_graph = build_nav_graph(
            kps,
            kps[:, 2],  # 高度 = z 坐标
            self.connection_radius,
            self.max_height_change,
            self.max_slope_angle,
        )
        self.nav_graph.traversable = traversable

        self.get_logger().info(
            f'收到导航图: {len(kps)} 关键点, '
            f'{sum(len(v) for v in self.nav_graph.adjacency.values())} 条边'
        )

    def _on_odom(self, msg: Odometry) -> None:
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.get_logger().info(
            f'收到目标: ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})'
        )

    def _plan_tick(self) -> None:
        if self.nav_graph is None or self.goal is None:
            return

        if self.nav_graph.tree is None or len(self.nav_graph.keypoints) == 0:
            return

        dist = np.sqrt(
            (self.current_x - self.goal[0]) ** 2
            + (self.current_y - self.goal[1]) ** 2
            + (self.current_z - self.goal[2]) ** 2
        )
        if dist < self.goal_tolerance:
            self.get_logger().info('已到达目标')
            self.goal = None
            return

        path_world = self.plan(
            (self.current_x, self.current_y, self.current_z), self.goal
        )
        if path_world is None:
            self.get_logger().warn('Dijkstra 未找到路径')
            return

        if len(path_world) > 2:
            path_world = smooth_path_3d(path_world, self.smooth_iters)

        self._publish_path(path_world)

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> Optional[List[Tuple[float, float, float]]]:
        """规划 3D 路径。返回 [(x, y, z), ...] 或 None。"""
        if self.nav_graph is None or self.nav_graph.tree is None:
            return None

        tree = self.nav_graph.tree
        kps = self.nav_graph.keypoints

        # 找最近可通行关键点
        start_idx = find_nearest_keypoint(tree, start)
        goal_idx = find_nearest_keypoint(tree, goal)

        # 检查可通行性
        if not self.nav_graph.traversable[start_idx]:
            self.get_logger().warn(f'起点关键点 {start_idx} 不可通行')
            return None
        if not self.nav_graph.traversable[goal_idx]:
            self.get_logger().warn(f'终点关键点 {goal_idx} 不可通行')
            return None

        # Dijkstra 搜索
        path_indices = dijkstra_search(
            self.nav_graph.adjacency, start_idx, goal_idx
        )
        if path_indices is None:
            return None

        # 转换为世界坐标并插值
        path_world = graph_path_to_world(
            path_indices, kps, self.interp_points
        )

        return path_world

    def _publish_path(
        self, path_world: List[Tuple[float, float, float]]
    ) -> None:
        """发布带 Z 坐标的路径。"""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for x, y, z in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0
            )
            msg.poses.append(pose)
        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
