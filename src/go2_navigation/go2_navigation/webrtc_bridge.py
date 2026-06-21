"""WebRTC Bridge：将 ROS2 /cmd_vel 速度指令转发到 Go2 机器人。

通过 unitree_webrtc_connect 的 Go2WebRTCClient 建立连接，
订阅 /cmd_vel 并转换为 WebRTC Move 指令发送。
超时自动停止。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

import asyncio
import threading
import logging

logger = logging.getLogger(__name__)


class Go2WebRTCClient:
    """线程安全的 WebRTC 客户端封装。"""

    def __init__(self, ip: str, aes_key: str):
        self.ip = ip
        self.aes_key = aes_key
        self._conn = None
        self._loop = None
        self._thread = None
        self._connected = threading.Event()
        self._running = False

    def connect(self, timeout: float = 15.0) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._connected.wait(timeout=timeout):
            self._running = False
            return False
        return True

    def disconnect(self):
        self._running = False
        if self._loop and self._conn:
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_run())
        except Exception as e:
            logger.error(f'WebRTC loop error: {e}')
        finally:
            self._connected.set()

    async def _connect_and_run(self):
        try:
            self._conn = UnitreeWebRTCConnection(
                WebRTCConnectionMethod.LocalSTA,
                ip=self.ip,
                aes_128_key=self.aes_key,
            )
            await self._conn.connect()
            await self._conn.datachannel.disableTrafficSaving(True)
            self._connected.set()
            while self._running:
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f'WebRTC connection failed: {e}')
            self._connected.set()

    async def _cleanup(self):
        if self._conn:
            try:
                await self._conn.close()
            except Exception:
                pass

    def _submit(self, coro):
        if not self._loop or not self._running:
            return None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=2.0)
        except Exception as e:
            logger.warning(f'Async call failed: {e}')
            return None

    def send_move(self, vx: float, vy: float, vyaw: float):
        if not self._conn or not self._running:
            return

        async def _send():
            await self._conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC['SPORT_MOD'],
                {
                    'api_id': SPORT_CMD['Move'],
                    'parameter': {'x': vx, 'y': vy, 'z': vyaw},
                },
            )

        self._submit(_send())

    def send_command(self, command: str):
        if not self._conn or not self._running:
            return
        if command not in SPORT_CMD:
            logger.error(f'Unknown command: {command}')
            return

        async def _send():
            await self._conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC['SPORT_MOD'],
                {'api_id': SPORT_CMD[command]},
            )

        self._submit(_send())


class WebRTCBridge(Node):
    def __init__(self) -> None:
        super().__init__('webrtc_bridge')

        # ── 参数 ──
        self.declare_parameter('robot_ip', '192.168.12.1')
        self.declare_parameter('aes_key', '7b7ad05fae7b79f3c0135f7417f895d0')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)

        self.robot_ip = self.get_parameter('robot_ip').value
        self.aes_key = self.get_parameter('aes_key').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        self.max_lv = self.get_parameter('max_linear_vel').value
        self.max_av = self.get_parameter('max_angular_vel').value

        # ── 状态 ──
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vyaw = 0.0
        self._zero_sent = False

        # ── 连接机器人 ──
        self.get_logger().info(f'连接 Go2: {self.robot_ip}...')
        self.client = Go2WebRTCClient(self.robot_ip, self.aes_key)
        if not self.client.connect(timeout=15.0):
            self.get_logger().error('连接 Go2 失败!')
            return
        self.get_logger().info('Go2 WebRTC 连接成功')

        # ── 订阅 ──
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10,
        )
        self.create_subscription(Twist, cmd_topic, self._on_cmd_vel, cmd_qos)

        # ── 超时检测 ──
        self._last_cmd_time = self.get_clock().now()
        self.create_timer(0.2, self._check_timeout)

        self.get_logger().info(f'WebRTC Bridge 已启动，监听 {cmd_topic}')

    def _on_cmd_vel(self, msg: Twist) -> None:
        vx = max(-self.max_lv, min(self.max_lv, msg.linear.x))
        vy = max(-self.max_lv, min(self.max_lv, msg.linear.y))
        vyaw = max(-self.max_av, min(self.max_av, msg.angular.z))

        # 跳过无变化的指令
        if (
            abs(vx - self._last_vx) < 0.01
            and abs(vy - self._last_vy) < 0.01
            and abs(vyaw - self._last_vyaw) < 0.01
        ):
            return

        self._last_vx = vx
        self._last_vy = vy
        self._last_vyaw = vyaw
        self._last_cmd_time = self.get_clock().now()
        self._zero_sent = False

        self.client.send_move(vx, vy, vyaw)

    def _check_timeout(self) -> None:
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if elapsed > 0.5 and not self._zero_sent:
            self.client.send_move(0.0, 0.0, 0.0)
            self._last_vx = 0.0
            self._last_vy = 0.0
            self._last_vyaw = 0.0
            self._zero_sent = True

    def destroy_node(self):
        if hasattr(self, 'client'):
            self.client.send_move(0.0, 0.0, 0.0)
            self.client.send_command('StopMove')
            self.client.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebRTCBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
