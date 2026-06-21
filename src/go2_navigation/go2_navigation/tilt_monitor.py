"""监控 FAST-LIO 输出的姿态，诊断地图倾斜原因。

用法:
    ros2 run go2_navigation tilt_monitor
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def quaternion_to_rpy(x: float, y: float, z: float, w: float):
    """四元数转 Roll/Pitch/Yaw（弧度）。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class TiltMonitor(Node):
    def __init__(self) -> None:
        super().__init__('tilt_monitor')
        self.sub = self.create_subscription(
            Odometry, '/Odometry', self._on_odom, 10
        )
        self.get_logger().info('倾斜监控已启动，订阅 /Odometry')
        self.get_logger().info('格式: Roll(横滚) | Pitch(俯仰) | Yaw(航向)  单位:度')

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        self.get_logger().info(
            f'Roll={roll_deg:+7.2f}°  Pitch={pitch_deg:+7.2f}°  Yaw={yaw_deg:+7.2f}°'
        )

        if abs(roll_deg) > 5.0 or abs(pitch_deg) > 5.0:
            self.get_logger().warn(
                f'检测到明显倾斜！Roll={roll_deg:.1f}° Pitch={pitch_deg:.1f}°'
            )


def main(args=None):
    rclpy.init(args=args)
    node = TiltMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
