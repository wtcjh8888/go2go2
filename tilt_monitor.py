#!/usr/bin/env python3
"""监控 FAST-LIO 姿态，诊断地图倾斜。

用法：python3 tilt_monitor.py
日志自动保存到 ~/C206Go2/tilt_log.csv
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def quaternion_to_rpy(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class TiltMonitor(Node):
    def __init__(self):
        super().__init__('tilt_monitor')

        log_path = Path.home() / 'C206Go2' / 'tilt_log.csv'
        self.csv_file = open(log_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['time', 'roll_deg', 'pitch_deg', 'yaw_deg',
                              'pos_x', 'pos_y', 'pos_z'])
        self.start_time = time.time()

        self.create_subscription(Odometry, '/Odometry', self._cb, 10)
        self.get_logger().info(f'监控中... 日志保存到 {log_path}')
        self.get_logger().info('Ctrl+C 退出')

    def _cb(self, msg):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        p = msg.pose.pose.position
        t = time.time() - self.start_time

        self.writer.writerow([f'{t:.2f}', f'{roll:.2f}', f'{pitch:.2f}',
                              f'{yaw:.2f}', f'{p.x:.3f}', f'{p.y:.3f}', f'{p.z:.3f}'])

        warn = '  <<< 倾斜!' if abs(roll) > 5 or abs(pitch) > 5 else ''
        self.get_logger().info(
            f't={t:7.1f}s  Roll={roll:+7.2f}°  Pitch={pitch:+7.2f}°  '
            f'Z={p.z:.2f}m{warn}'
        )

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


rclpy.init()
node = TiltMonitor()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
