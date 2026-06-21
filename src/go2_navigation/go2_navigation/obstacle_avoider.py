"""3D 感知的实时避障模块：基于 VFH + 高度过滤。

订阅实时点云和 Odometry，使用 VFH 算法计算最佳通行方向，
输出修正后的速度指令。支持高度带过滤，区分地面/障碍物/坡道。
"""

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2


# ── 核心算法（可独立测试）──────────────────────────────────────


def build_vfh_histogram(
    dists: np.ndarray,
    angles: np.ndarray,
    sector_count: int = 72,
    detection_range: float = 2.0,
) -> np.ndarray:
    """构建 VFH 直方图：每个扇区存储最近障碍物距离。"""
    sector_angle = 360.0 / sector_count
    histogram = np.full(sector_count, detection_range)

    for dist, angle in zip(dists, angles):
        angle_deg = math.degrees(angle) % 360.0
        sector = int(angle_deg / sector_angle) % sector_count
        histogram[sector] = min(histogram[sector], dist)

    return histogram


def find_best_direction(
    histogram: np.ndarray,
    warning_dist: float = 0.6,
    sector_count: int = 72,
) -> Optional[float]:
    """找到最佳通行方向（正前方 ±90° 范围内最通畅的扇区）。"""
    sector_angle = 360.0 / sector_count

    best_sector = None
    best_cost = float('inf')

    for offset in range(-18, 19):  # ±90°
        sector = offset % sector_count
        if histogram[sector] >= warning_dist:
            cost = abs(offset)
            if cost < best_cost:
                best_cost = cost
                best_sector = sector

    if best_sector is None:
        return None

    return best_sector * sector_angle + sector_angle / 2.0


def check_danger_ahead_3d(
    points: np.ndarray,
    robot_z: float,
    danger_dist: float = 0.3,
    fov_angle: float = 120.0,
    height_band: float = 0.5,
    ground_clearance: float = 0.05,
) -> bool:
    """3D 感知的前方危险检测。

    只考虑 robot_z 附近的点（忽略地面和头顶）。

    Args:
        points: (N, 3) 点云（机体坐标系）
        robot_z: 机器人当前 Z 坐标
        danger_dist: 危险距离 (m)
        fov_angle: 前方检测角度 (度)
        height_band: 高度带（±robot_z 范围）
        ground_clearance: 地面间隙（忽略 robot_z 以下的点）

    Returns:
        True 表示前方有危险障碍物
    """
    if len(points) == 0:
        return False

    # 高度过滤：只看 robot_z 附近的点
    z_mask = (points[:, 2] >= -ground_clearance) & (points[:, 2] <= height_band)
    filtered = points[z_mask]
    if len(filtered) == 0:
        return False

    dists = np.sqrt(filtered[:, 0] ** 2 + filtered[:, 1] ** 2)
    angles = np.abs(np.arctan2(filtered[:, 1], filtered[:, 0]))
    fov_half = math.radians(fov_angle / 2.0)
    mask = (dists < danger_dist) & (angles < fov_half)
    return bool(np.any(mask))


def filter_fov_points_3d(
    points: np.ndarray,
    robot_z: float,
    detection_range: float = 2.0,
    fov_angle: float = 120.0,
    height_band: float = 0.5,
    ground_clearance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """3D 感知的点云过滤：高度带 + FOV + 距离。

    Args:
        points: (N, 3) 点云
        robot_z: 机器人 Z 坐标
        detection_range: 检测范围 (m)
        fov_angle: 前方检测角度 (度)
        height_band: 高度带上限（相对 robot_z）
        ground_clearance: 地面间隙（相对 robot_z 向下）

    Returns:
        (dists, angles) 过滤后的距离和角度数组
    """
    if len(points) == 0:
        return np.array([]), np.array([])

    # 高度过滤
    z_mask = (points[:, 2] >= -ground_clearance) & (points[:, 2] <= height_band)
    filtered = points[z_mask]
    if len(filtered) == 0:
        return np.array([]), np.array([])

    dists = np.sqrt(filtered[:, 0] ** 2 + filtered[:, 1] ** 2)
    angles = np.arctan2(filtered[:, 1], filtered[:, 0])
    fov_half = math.radians(fov_angle / 2.0)

    mask = (dists < detection_range) & (np.abs(angles) < fov_half)
    return dists[mask], angles[mask]


def classify_terrain(
    points: np.ndarray,
    robot_z: float,
    forward_range: float = 1.5,
    slope_threshold: float = 30.0,
) -> Tuple[bool, float]:
    """判断前方地形是坡道还是平地。

    拟合前方点云的平面，计算坡度角。

    Args:
        points: (N, 3) 点云（机体坐标系）
        robot_z: 机器人 Z 坐标
        forward_range: 前方检测范围 (m)
        slope_threshold: 坡度阈值（度）

    Returns:
        (is_slope, slope_angle_degrees)
    """
    if len(points) < 10:
        return False, 0.0

    # 只看前方、robot_z 附近的点
    forward_mask = (points[:, 0] > 0) & (points[:, 0] < forward_range)
    z_mask = (points[:, 2] >= -0.3) & (points[:, 2] <= 1.0)
    filtered = points[forward_mask & z_mask]

    if len(filtered) < 10:
        return False, 0.0

    # 简单平面拟合：用 z 和 x 的线性关系估算坡度
    x_vals = filtered[:, 0]
    z_vals = filtered[:, 2]

    # 最小二乘拟合 z = a*x + b
    try:
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        result = np.linalg.lstsq(A, z_vals, rcond=None)
        slope = result[0][0]  # dz/dx
        slope_angle = math.degrees(math.atan(abs(slope)))
        return slope_angle > slope_threshold, slope_angle
    except Exception:
        return False, 0.0


def compute_avoidance_cmd(
    best_angle_deg: float,
    raw_vx: float = 0.0,
    max_lin: float = 0.2,
    max_ang: float = 0.5,
) -> Tuple[float, float]:
    """根据最佳通行方向计算避障速度指令。"""
    angle_rad = math.radians(best_angle_deg)

    angle_factor = max(0.0, 1.0 - abs(angle_rad) / math.pi)
    vx = raw_vx * angle_factor * 0.5

    vyaw = max(-max_ang, min(max_ang, angle_rad * 2.0))

    return vx, vyaw


# ── ROS2 节点 ─────────────────────────────────────────────────


class ObstacleAvoider(Node):
    def __init__(self) -> None:
        super().__init__('obstacle_avoider')

        self.declare_parameter('cloud_topic', '/cloud_registered_body')
        self.declare_parameter('odom_topic', '/Odometry_base_link')
        self.declare_parameter('detection_range', 2.0)
        self.declare_parameter('danger_distance', 0.3)
        self.declare_parameter('warning_distance', 0.6)
        self.declare_parameter('fov_angle', 120.0)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('height_band', 0.5)
        self.declare_parameter('ground_clearance', 0.05)
        self.declare_parameter('slope_threshold', 30.0)
        self.declare_parameter('input_cmd_vel_topic', '/cmd_vel_raw')
        self.declare_parameter('output_cmd_vel_topic', '/cmd_vel')

        cloud_topic = self.get_parameter('cloud_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.detection_range = self.get_parameter('detection_range').value
        self.danger_dist = self.get_parameter('danger_distance').value
        self.warning_dist = self.get_parameter('warning_distance').value
        self.fov_angle = self.get_parameter('fov_angle').value
        self.max_lin = self.get_parameter('max_linear_speed').value
        self.max_ang = self.get_parameter('max_angular_speed').value
        self.height_band = self.get_parameter('height_band').value
        self.ground_clearance = self.get_parameter('ground_clearance').value
        self.slope_threshold = self.get_parameter('slope_threshold').value
        input_topic = self.get_parameter('input_cmd_vel_topic').value
        output_topic = self.get_parameter('output_cmd_vel_topic').value

        self._points: Optional[np.ndarray] = None
        self._robot_z: float = 0.0
        self._raw_cmd = Twist()
        self._sector_count = 72

        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        self.create_subscription(
            PointCloud2, cloud_topic, self._on_cloud, cloud_qos
        )
        self.create_subscription(Odometry, odom_topic, self._on_odom, odom_qos)
        self.create_subscription(Twist, input_topic, self._on_cmd_vel, 10)
        self.cmd_pub = self.create_publisher(Twist, output_topic, 10)
        self.create_timer(0.05, self._control_tick)  # 20Hz

        self.get_logger().info(
            f'3D 避障模块已启动: FOV={self.fov_angle}°, '
            f'高度带={self.height_band}m, 地面间隙={self.ground_clearance}m'
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        points = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        )
        arr = np.asarray(points)
        if arr.size == 0:
            return
        if arr.dtype.names:
            self._points = np.column_stack(
                [arr['x'], arr['y'], arr['z']]
            ).astype(np.float32)
        else:
            self._points = arr.reshape(-1, 3).astype(np.float32)

    def _on_odom(self, msg: Odometry) -> None:
        self._robot_z = msg.pose.pose.position.z

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._raw_cmd = msg

    def _control_tick(self) -> None:
        if self._points is None:
            self.cmd_pub.publish(self._raw_cmd)
            return

        # 地形分类：如果是坡道，跳过避障（允许通过）
        is_slope, slope_angle = classify_terrain(
            self._points, self._robot_z, slope_threshold=self.slope_threshold
        )

        if is_slope:
            # 坡道上不避障，只做速度衰减
            cmd = Twist()
            cmd.linear.x = self._raw_cmd.linear.x * max(0.3, 1.0 - slope_angle / 90.0)
            cmd.angular.z = self._raw_cmd.angular.z
            self.cmd_pub.publish(cmd)
            return

        # 3D 危险检测
        if check_danger_ahead_3d(
            self._points,
            self._robot_z,
            self.danger_dist,
            self.fov_angle,
            self.height_band,
            self.ground_clearance,
        ):
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.get_logger().warn('紧急停止: 前方有危险障碍物 (3D)')
            return

        # 3D VFH 避障
        dists, angles = filter_fov_points_3d(
            self._points,
            self._robot_z,
            self.detection_range,
            self.fov_angle,
            self.height_band,
            self.ground_clearance,
        )

        if len(dists) == 0:
            self.cmd_pub.publish(self._raw_cmd)
            return

        histogram = build_vfh_histogram(
            dists, angles, self._sector_count, self.detection_range
        )

        best_angle = find_best_direction(
            histogram, self.warning_dist, self._sector_count
        )

        if best_angle is None:
            # 没有通畅方向，减速
            cmd = Twist()
            cmd.linear.x = self._raw_cmd.linear.x * 0.3
            cmd.angular.z = self._raw_cmd.angular.z
            self.cmd_pub.publish(cmd)
            return

        # 检查正前方是否通畅
        sector_angle = 360.0 / self._sector_count
        forward_sector = int(0.0 / sector_angle) % self._sector_count
        if histogram[forward_sector] >= self.warning_dist:
            self.cmd_pub.publish(self._raw_cmd)
        else:
            vx, vyaw = compute_avoidance_cmd(
                best_angle, self._raw_cmd.linear.x, self.max_lin, self.max_ang
            )
            cmd = Twist()
            cmd.linear.x = vx
            cmd.angular.z = vyaw
            self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
