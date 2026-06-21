"""手动位姿节点：不依赖 LiDAR，通过 RViz 设置初始位置。

功能：
1. 订阅 /initial_pose（RViz 的 "2D Pose Estimate" 按钮）
2. 发布 /Odometry（固定在设定的位置，不移动）
3. 支持通过参数直接设置初始位置

用法：
    ros2 run go2_navigation manual_pose
    ros2 run go2_navigation manual_pose --ros-args -p x:=1.0 -p y:=2.0

    # 在 RViz 中用 "2D Pose Estimate" 按钮点击设置位置
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
)
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """yaw 角 → 四元数。"""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ManualPose(Node):
    def __init__(self) -> None:
        super().__init__('manual_pose')

        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('initial_pose_topic', '/initial_pose')

        self._x = self.get_parameter('x').value
        self._y = self.get_parameter('y').value
        self._yaw = self.get_parameter('yaw').value
        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('initial_pose_topic').value

        # 订阅 RViz 的 "2D Pose Estimate"
        self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self._on_pose, 10
        )

        # 发布 /Odometry 和 TF
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 10Hz 发布
        self.create_timer(0.1, self._publish_odom)

        self.get_logger().info(
            f'手动位姿节点已启动: 初始位置=({self._x:.2f}, {self._y:.2f}, {self._yaw:.2f}rad)'
        )
        self.get_logger().info('在 RViz 中用 "2D Pose Estimate" 按钮可修改位置')

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """收到 RViz 的位姿设置。"""
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # 四元数 → yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny_cosp, cosy_cosp)

        self.get_logger().info(
            f'位置已更新: ({self._x:.2f}, {self._y:.2f}, {math.degrees(self._yaw):.1f}°)'
        )

    def _publish_odom(self) -> None:
        """发布固定的 Odometry。"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = yaw_to_quaternion(self._yaw)

        # 速度为 0
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(msg)

        # 发布 TF: map → base_link
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quaternion(self._yaw)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ManualPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
