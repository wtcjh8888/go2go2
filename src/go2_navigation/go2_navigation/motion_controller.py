"""平面运动控制器：在 3D 地图路径上执行 XY Pure Pursuit 路径跟踪。

订阅 /planned_path 和补偿后的 base_link Odometry，发布 /cmd_vel_raw 给避障模块。
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist


# ── 核心算法（可独立测试）──────────────────────────────────────


def euler_from_quaternion(q: Tuple[float, float, float, float]) -> float:
    """四元数 → yaw 角 (弧度)。"""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def pure_pursuit_control(
    robot_x: float,
    robot_y: float,
    robot_z: float,
    robot_yaw: float,
    target_x: float,
    target_y: float,
    target_z: float,
    max_lin: float = 0.2,
    max_ang: float = 0.5,
    min_lin: float = 0.0,
    min_ang: float = 0.0,
    max_slope_angle: float = 30.0,
    slope_speed_factor: float = 0.5,
) -> Tuple[float, float]:
    """3D 感知的 Pure Pursuit 控制律。

    XY 控制逻辑与 2D 版本相同，新增坡度减速：
    根据目标点与机器人的高差和水平距离计算坡度，按比例降低线速度。

    Returns:
        (vx, vyaw) 速度指令
    """
    dx = target_x - robot_x
    dy = target_y - robot_y
    dz = target_z - robot_z
    dist = np.hypot(dx, dy)
    if dist < 0.01:
        return 0.0, 0.0

    # 目标在机器人坐标系下的位置
    local_x = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
    local_y = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

    # 曲率
    curvature = 2.0 * local_y / (dist * dist)

    # 线速度：根据角度差调整
    angle_to_target = math.atan2(local_y, local_x)
    angle_factor = max(0.0, math.cos(angle_to_target))
    vx = max_lin * angle_factor
    if vx > 1e-6 and vx < min_lin:
        vx = min(min_lin, max_lin)

    # 坡度减速
    if dist > 0.01:
        slope_deg = math.degrees(math.atan2(abs(dz), dist))
        slope_scale = 1.0 - slope_speed_factor * min(
            slope_deg / max(max_slope_angle, 1.0), 1.0
        )
        vx *= max(0.3, slope_scale)  # 最低保留 30% 速度

    # 角速度
    # 角速度不要依赖线速度，否则目标在侧前方时 Go2 只会给出极小角速度。
    vyaw = max(-max_ang, min(max_ang, angle_to_target * 1.5))
    if abs(vyaw) > 1e-6 and abs(vyaw) < min_ang:
        vyaw = math.copysign(min(min_ang, max_ang), vyaw)

    return vx, vyaw


def smooth_velocity(
    target: float, current: float, dt: float, max_acc: float = 0.3
) -> float:
    """限制加速度，防止速度突变。"""
    max_change = max_acc * dt
    change = target - current
    if abs(change) > max_change:
        change = math.copysign(max_change, change)
    return current + change


def find_look_ahead_point(
    path: List[Tuple[float, float, float]],
    robot_x: float,
    robot_y: float,
    robot_z: float,
    start_idx: int,
    look_ahead: float = 0.3,
) -> Optional[Tuple[float, float, float]]:
    """从 start_idx 开始，按 XY 平面距离找到前视点。"""
    for i in range(start_idx, len(path)):
        px, py, pz = path[i]
        dist = np.hypot(robot_x - px, robot_y - py)
        if dist >= look_ahead:
            return (px, py, pz)
    return path[-1] if path else None


# ── ROS2 节点 ─────────────────────────────────────────────────


class MotionController(Node):
    def __init__(self) -> None:
        super().__init__('motion_controller')

        self.declare_parameter('look_ahead_distance', 0.3)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('min_linear_speed', 0.0)
        self.declare_parameter('min_angular_speed', 0.0)
        self.declare_parameter('max_acceleration', 0.3)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('max_slope_angle', 30.0)
        self.declare_parameter('slope_speed_factor', 0.5)
        self.declare_parameter('odom_topic', '/Odometry_base_link')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_raw')

        self.look_ahead = self.get_parameter('look_ahead_distance').value
        ctrl_freq = self.get_parameter('control_frequency').value
        self.max_lin = self.get_parameter('max_linear_speed').value
        self.max_ang = self.get_parameter('max_angular_speed').value
        self.min_lin = self.get_parameter('min_linear_speed').value
        self.min_ang = self.get_parameter('min_angular_speed').value
        self.max_acc = self.get_parameter('max_acceleration').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.max_slope_angle = self.get_parameter('max_slope_angle').value
        self.slope_speed_factor = self.get_parameter('slope_speed_factor').value
        odom_topic = self.get_parameter('odom_topic').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value

        self._path: List[Tuple[float, float, float]] = []
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._yaw = 0.0
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0
        self._last_time = self.get_clock().now()

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        self.create_subscription(Odometry, odom_topic, self._on_odom, odom_qos)
        self.create_subscription(Path, '/planned_path', self._on_path, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(1.0 / ctrl_freq, self._control_tick)

        self.get_logger().info(
            f'平面运动控制器已启动: 前视={self.look_ahead}m, '
            f'最大速度={self.max_lin}m/s, 里程计={odom_topic}'
        )

    def _on_odom(self, msg: Odometry) -> None:
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        self._yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def _on_path(self, msg: Path) -> None:
        self._path = [
            (p.pose.position.x, p.pose.position.y, p.pose.position.z)
            for p in msg.poses
        ]
        self.get_logger().info(f'收到路径: {len(self._path)} 个点')

    def _control_tick(self) -> None:
        if not self._path:
            return

        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now
        dt = max(dt, 0.001)

        # 找最近点（XY 平面距离）
        min_idx = min(
            range(len(self._path)),
            key=lambda i: np.sqrt(
                (self._x - self._path[i][0]) ** 2
                + (self._y - self._path[i][1]) ** 2
            ),
        )

        # 检查是否到达目标（XY 平面距离）
        goal_x, goal_y, goal_z = self._path[-1]
        goal_dist = np.hypot(self._x - goal_x, self._y - goal_y)
        if goal_dist < self.goal_tol:
            self._publish_stop()
            self.get_logger().info('到达目标，停止')
            return

        # 找前视点
        target = find_look_ahead_point(
            self._path, self._x, self._y, self._z, min_idx, self.look_ahead
        )
        if target is None:
            target = self._path[-1]

        # Pure Pursuit（XY 平面控制，路径 Z 只保留作地图高度信息）
        vx, vyaw = pure_pursuit_control(
            self._x,
            self._y,
            self._z,
            self._yaw,
            target[0],
            target[1],
            target[2],
            self.max_lin,
            self.max_ang,
            self.min_lin,
            self.min_ang,
            self.max_slope_angle,
            self.slope_speed_factor,
        )

        # 速度平滑
        vx = smooth_velocity(vx, self._prev_vx, dt, self.max_acc)
        vyaw = smooth_velocity(vyaw, self._prev_vyaw, dt, self.max_acc)

        self._prev_vx = vx
        self._prev_vyaw = vyaw

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = vyaw
        self.cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0
        self._path = []


def main(args=None):
    rclpy.init(args=args)
    node = MotionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
