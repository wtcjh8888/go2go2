"""Republish FAST-LIO body odometry as base_link odometry.

FAST-LIO publishes the LiDAR/IMU body pose. On the Go2 the MID360S is mounted
with a pitch angle, so body is not the horizontal robot base. This node applies
the static TF body -> base_link and publishes an odometry message for base_link
in the same world frame.
"""

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener


QuaternionTuple = Tuple[float, float, float, float]
VectorTuple = Tuple[float, float, float]


def _normalize_quaternion(q: QuaternionTuple) -> QuaternionTuple:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def _quaternion_multiply(
    q1: QuaternionTuple, q2: QuaternionTuple
) -> QuaternionTuple:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return _normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def _rotate_vector(q: QuaternionTuple, v: VectorTuple) -> VectorTuple:
    x, y, z, w = _normalize_quaternion(q)
    vx, vy, vz = v

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


class OdometryToBaseLink(Node):
    def __init__(self) -> None:
        super().__init__('odometry_to_base_link')

        self.declare_parameter('input_odom_topic', '/Odometry')
        self.declare_parameter('output_odom_topic', '/Odometry_base_link')
        self.declare_parameter('body_frame', 'body')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tf_timeout', 0.2)

        input_topic = self.get_parameter('input_odom_topic').value
        output_topic = self.get_parameter('output_odom_topic').value
        self.body_frame = self.get_parameter('body_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.tf_timeout = float(self.get_parameter('tf_timeout').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.body_to_base: Optional[TransformStamped] = None

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=20)
        self.odom_pub = self.create_publisher(Odometry, output_topic, qos)
        self.create_subscription(Odometry, input_topic, self._on_odom, qos)

        self.get_logger().info(
            f'Odometry converter: {input_topic} ({self.body_frame}) -> '
            f'{output_topic} ({self.base_frame})'
        )

    def _lookup_body_to_base(self) -> Optional[TransformStamped]:
        if self.body_to_base is not None:
            return self.body_to_base

        try:
            self.body_to_base = self.tf_buffer.lookup_transform(
                self.body_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout),
            )
            t = self.body_to_base.transform.translation
            q = self.body_to_base.transform.rotation
            self.get_logger().info(
                f'Got TF {self.body_frame} -> {self.base_frame}: '
                f't=({t.x:.3f}, {t.y:.3f}, {t.z:.3f}), '
                f'q=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Waiting for TF {self.body_frame} -> {self.base_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        return self.body_to_base

    def _on_odom(self, msg: Odometry) -> None:
        body_to_base = self._lookup_body_to_base()
        if body_to_base is None:
            return

        body_pos = msg.pose.pose.position
        body_q_msg = msg.pose.pose.orientation
        body_q = _normalize_quaternion(
            (body_q_msg.x, body_q_msg.y, body_q_msg.z, body_q_msg.w)
        )

        tf_t = body_to_base.transform.translation
        tf_q_msg = body_to_base.transform.rotation
        base_offset = (tf_t.x, tf_t.y, tf_t.z)
        body_to_base_q = _normalize_quaternion(
            (tf_q_msg.x, tf_q_msg.y, tf_q_msg.z, tf_q_msg.w)
        )

        rotated_offset = _rotate_vector(body_q, base_offset)
        base_q = _quaternion_multiply(body_q, body_to_base_q)

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = self.base_frame
        out.pose.pose.position.x = body_pos.x + rotated_offset[0]
        out.pose.pose.position.y = body_pos.y + rotated_offset[1]
        out.pose.pose.position.z = body_pos.z + rotated_offset[2]
        out.pose.pose.orientation.x = base_q[0]
        out.pose.pose.orientation.y = base_q[1]
        out.pose.pose.orientation.z = base_q[2]
        out.pose.pose.orientation.w = base_q[3]
        out.pose.covariance = msg.pose.covariance
        out.twist = msg.twist

        self.odom_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryToBaseLink()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
