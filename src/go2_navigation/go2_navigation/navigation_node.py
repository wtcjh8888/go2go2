"""导航主节点：协调路径规划、避障、运动控制，实现端到端自主导航。

状态机：IDLE → PLANNING → FOLLOWING → GOAL_REACHED
                       → OBSTACLE_AVOID → FOLLOWING
                       → REPLANNING → FOLLOWING
"""

import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
import math as _math


def __euler_from_quaternion(q):
    """四元数 → 欧拉角 (roll, pitch, yaw)。"""
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = _math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = _math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = _math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class NavState(Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    FOLLOWING = 'following'
    OBSTACLE_AVOID = 'obstacle_avoid'
    REPLANNING = 'replanning'
    GOAL_REACHED = 'goal_reached'


class NavigationNode(Node):
    def __init__(self) -> None:
        super().__init__('navigation_node')

        # ── 参数 ──
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('replan_distance', 1.0)
        self.declare_parameter('replan_interval', 5.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('planner_frequency', 2.0)
        self.declare_parameter('smooth_iterations', 3)
        self.declare_parameter('look_ahead_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('max_acceleration', 0.5)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('cloud_topic', '/cloud_registered_body')
        self.declare_parameter('detection_range', 3.0)
        self.declare_parameter('danger_distance', 0.3)
        self.declare_parameter('warning_distance', 0.8)
        self.declare_parameter('fov_angle', 120.0)

        odom_topic = self.get_parameter('odom_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        self.replan_dist = self.get_parameter('replan_distance').value
        self.replan_interval = self.get_parameter('replan_interval').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.planner_freq = self.get_parameter('planner_frequency').value
        self.smooth_iters = self.get_parameter('smooth_iterations').value
        self.look_ahead = self.get_parameter('look_ahead_distance').value
        self.max_lin = self.get_parameter('max_linear_speed').value
        self.max_ang = self.get_parameter('max_angular_speed').value
        self.max_acc = self.get_parameter('max_acceleration').value
        ctrl_freq = self.get_parameter('control_frequency').value
        cloud_topic = self.get_parameter('cloud_topic').value
        self.detection_range = self.get_parameter('detection_range').value
        self.danger_dist = self.get_parameter('danger_distance').value
        self.warning_dist = self.get_parameter('warning_distance').value
        self.fov_angle = self.get_parameter('fov_angle').value

        # ── 状态 ──
        self.state = NavState.IDLE
        self.grid: Optional[np.ndarray] = None
        self.grid_resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.grid_rows = 0
        self.grid_cols = 0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.goal: Optional[Tuple[float, float]] = None
        self.path: List[Tuple[float, float]] = []
        self.cloud_points: Optional[np.ndarray] = None
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0
        self._last_plan_time = self.get_clock().now()
        self._last_time = self.get_clock().now()

        # VFH
        self._sector_count = 72
        self._sector_angle = 360.0 / self._sector_count

        # ── QoS ──
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )
        cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=5,
        )

        # ── 订阅 ──
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, odom_qos)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2 as pc2_mod
        self._pc2_mod = pc2_mod
        self.create_subscription(
            PointCloud2, cloud_topic, self._on_cloud, cloud_qos
        )

        # ── 发布 ──
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        # ── 控制循环 ──
        self.create_timer(1.0 / ctrl_freq, self._control_tick)

        self.get_logger().info(
            f'导航主节点已启动，状态={self.state.value}'
        )

    # ── 回调 ────────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.grid_resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.grid_cols = msg.info.width
        self.grid_rows = msg.info.height
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            (self.grid_rows, self.grid_cols)
        )
        self.get_logger().info(
            f'收到地图: {self.grid_cols}x{self.grid_rows}'
        )

    def _on_odom(self, msg: Odometry) -> None:
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = _euler_from_quaternion([q.x, q.y, q.z, q.w])

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f'收到目标: ({self.goal[0]:.2f}, {self.goal[1]:.2f})'
        )
        self.state = NavState.PLANNING

    def _on_cloud(self, msg) -> None:
        points = []
        for p in self._pc2_mod.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nan=True
        ):
            points.append(p)
        if points:
            self.cloud_points = np.array(points, dtype=np.float32)

    # ── 控制主循环 ──────────────────────────────────────────────

    def _control_tick(self) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now
        dt = max(dt, 0.001)

        if self.state == NavState.IDLE:
            return

        if self.state == NavState.PLANNING:
            self._do_planning()
            return

        if self.state == NavState.FOLLOWING:
            self._do_following(dt)
            return

        if self.state == NavState.OBSTACLE_AVOID:
            self._do_obstacle_avoid(dt)
            return

        if self.state == NavState.REPLANNING:
            self._do_replanning()
            return

        if self.state == NavState.GOAL_REACHED:
            self._publish_stop()
            self.state = NavState.IDLE
            return

    # ── 状态处理 ────────────────────────────────────────────────

    def _do_planning(self) -> None:
        """执行路径规划。"""
        if self.grid is None:
            self.get_logger().warn('等待地图...')
            return

        path = self._astar(
            (self.current_x, self.current_y), self.goal
        )
        if path is None:
            self.get_logger().warn('A* 未找到路径，重试...')
            return

        if len(path) > 2:
            path = self._smooth(path, self.smooth_iters)

        self.path = path
        self._publish_path(path)
        self._last_plan_time = self.get_clock().now()
        self.state = NavState.FOLLOWING
        self.get_logger().info(
            f'规划完成: {len(path)} 个点，进入路径跟踪'
        )

    def _do_following(self, dt: float) -> None:
        """Pure Pursuit 路径跟踪 + 避障检查。"""
        if not self.path:
            self.state = NavState.IDLE
            return

        # 检查是否到达目标
        goal_x, goal_y = self.path[-1]
        goal_dist = np.hypot(self.current_x - goal_x, self.current_y - goal_y)
        if goal_dist < self.goal_tol:
            self.get_logger().info('到达目标!')
            self.state = NavState.GOAL_REACHED
            return

        # 避障检查
        if self._check_obstacle_ahead():
            self.get_logger().warn('前方有障碍，进入避障')
            self.state = NavState.OBSTACLE_AVOID
            return

        # 检查是否偏离路径太远
        min_dist = min(
            np.hypot(self.current_x - px, self.current_y - py)
            for px, py in self.path
        )
        if min_dist > self.replan_dist:
            elapsed = (
                (self.get_clock().now() - self._last_plan_time).nanoseconds / 1e9
            )
            if elapsed > self.replan_interval:
                self.get_logger().warn('偏离路径，重新规划')
                self.state = NavState.REPLANNING
                return

        # Pure Pursuit
        min_idx = min(
            range(len(self.path)),
            key=lambda i: np.hypot(
                self.current_x - self.path[i][0],
                self.current_y - self.path[i][1],
            ),
        )

        target = self._find_look_ahead(min_idx)
        if target is None:
            target = self.path[-1]

        vx, vyaw = self._pure_pursuit(target)
        vx = self._smooth_vel(vx, self._prev_vx, dt)
        vyaw = self._smooth_vel(vyaw, self._prev_vyaw, dt)
        self._prev_vx = vx
        self._prev_vyaw = vyaw

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = vyaw
        self.cmd_pub.publish(cmd)

    def _do_obstacle_avoid(self, dt: float) -> None:
        """避障模式：VFH 转向。"""
        if not self._check_obstacle_ahead():
            self.get_logger().info('障碍已清除，恢复路径跟踪')
            self.state = NavState.FOLLOWING
            return

        if self.cloud_points is None:
            self._publish_stop()
            return

        # VFH 避障
        pts = self.cloud_points
        dists = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        in_range = dists < self.detection_range
        pts = pts[in_range]
        dists = dists[in_range]

        if len(pts) == 0:
            self._publish_stop()
            return

        angles = np.arctan2(pts[:, 1], pts[:, 0])
        fov_half = math.radians(self.fov_angle / 2.0)
        in_fov = np.abs(angles) < fov_half
        dists = dists[in_fov]
        angles = angles[in_fov]

        if len(dists) == 0:
            self._publish_stop()
            return

        # 构建直方图
        histogram = np.full(self._sector_count, self.detection_range)
        for d, a in zip(dists, angles):
            sector = int(math.degrees(a) % 360.0 / self._sector_angle)
            sector = sector % self._sector_count
            histogram[sector] = min(histogram[sector], d)

        # 找最佳方向
        target_sector = int(0.0 / self._sector_angle) % self._sector_count
        best_sector = target_sector
        best_cost = float('inf')
        for offset in range(-18, 19):
            sector = (target_sector + offset) % self._sector_count
            if histogram[sector] >= self.warning_dist:
                cost = abs(offset)
                if cost < best_cost:
                    best_cost = cost
                    best_sector = sector

        best_angle = best_sector * self._sector_angle + self._sector_angle / 2.0
        angle_rad = math.radians(best_angle)

        cmd = Twist()
        cmd.linear.x = self.max_lin * 0.3
        cmd.angular.z = max(-self.max_ang, min(self.max_ang, angle_rad * 2.0))
        self.cmd_pub.publish(cmd)

    def _do_replanning(self) -> None:
        """重新规划路径。"""
        self.state = NavState.PLANNING

    # ── 工具方法 ────────────────────────────────────────────────

    def _check_obstacle_ahead(self) -> bool:
        """检查前方是否有障碍物。"""
        if self.cloud_points is None:
            return False

        pts = self.cloud_points
        dists = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        angles = np.abs(np.arctan2(pts[:, 1], pts[:, 0]))
        fov_half = math.radians(self.fov_angle / 2.0)
        mask = (dists < self.danger_dist) & (angles < fov_half)
        return np.any(mask)

    def _find_look_ahead(
        self, start_idx: int
    ) -> Optional[Tuple[float, float]]:
        for i in range(start_idx, len(self.path)):
            px, py = self.path[i]
            dist = np.hypot(self.current_x - px, self.current_y - py)
            if dist >= self.look_ahead:
                return (px, py)
        return self.path[-1]

    def _pure_pursuit(
        self, target: Tuple[float, float]
    ) -> Tuple[float, float]:
        tx, ty = target
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = np.hypot(dx, dy)
        if dist < 0.01:
            return 0.0, 0.0

        local_x = dx * math.cos(self.current_yaw) + dy * math.sin(self.current_yaw)
        local_y = -dx * math.sin(self.current_yaw) + dy * math.cos(self.current_yaw)

        curvature = 2.0 * local_y / (dist * dist)
        angle_to_target = math.atan2(local_y, local_x)
        angle_factor = max(0.0, math.cos(angle_to_target))
        vx = self.max_lin * angle_factor
        vyaw = max(-self.max_ang, min(self.max_ang, curvature * vx * 2.0))
        return vx, vyaw

    def _smooth_vel(self, target: float, current: float, dt: float) -> float:
        max_change = self.max_acc * dt
        change = target - current
        if abs(change) > max_change:
            change = math.copysign(max_change, change)
        return current + change

    # ── A* ──────────────────────────────────────────────────────

    def _astar(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        if self.grid is None:
            return None

        sc = int((start[0] - self.origin_x) / self.grid_resolution)
        sr = int((start[1] - self.origin_y) / self.grid_resolution)
        gc = int((goal[0] - self.origin_x) / self.grid_resolution)
        gr = int((goal[1] - self.origin_y) / self.grid_resolution)

        if not (0 <= sr < self.grid_rows and 0 <= sc < self.grid_cols):
            return None
        if not (0 <= gr < self.grid_rows and 0 <= gc < self.grid_cols):
            return None
        if self.grid[sr, sc] == 100 or self.grid[gr, gc] == 100:
            return None

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        import heapq
        open_set = []
        heapq.heappush(open_set, (0.0, sr, sc))
        came_from = {}
        g_score = np.full((self.grid_rows, self.grid_cols), np.inf)
        g_score[sr, sc] = 0.0

        while open_set:
            _, r, c = heapq.heappop(open_set)
            if r == gr and c == gc:
                path = []
                while (r, c) in came_from:
                    col, row = c, r
                    path.append((
                        self.origin_x + (col + 0.5) * self.grid_resolution,
                        self.origin_y + (row + 0.5) * self.grid_resolution,
                    ))
                    r, c = came_from[(r, c)]
                col, row = c, r
                path.append((
                    self.origin_x + (col + 0.5) * self.grid_resolution,
                    self.origin_y + (row + 0.5) * self.grid_resolution,
                ))
                path.reverse()
                return path

            for dr, dc, cost in neighbors:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols):
                    continue
                if self.grid[nr, nc] == 100:
                    continue
                penalty = 5.0 if self.grid[nr, nc] == 50 else 0.0
                new_g = g_score[r, c] + cost + penalty
                if new_g < g_score[nr, nc]:
                    g_score[nr, nc] = new_g
                    came_from[(nr, nc)] = (r, c)
                    h = np.hypot(nr - gr, nc - gc)
                    heapq.heappush(open_set, (new_g + h, nr, nc))

        return None

    def _smooth(
        self, path: List[Tuple[float, float]], iterations: int
    ) -> List[Tuple[float, float]]:
        pts = [list(p) for p in path]
        for _ in range(iterations):
            new_pts = [pts[0]]
            for i in range(1, len(pts) - 1):
                new_pts.append([
                    0.25 * pts[i - 1][0] + 0.5 * pts[i][0] + 0.25 * pts[i + 1][0],
                    0.25 * pts[i - 1][1] + 0.5 * pts[i][1] + 0.25 * pts[i + 1][1],
                ])
            new_pts.append(pts[-1])
            pts = new_pts
        return [(p[0], p[1]) for p in pts]

    # ── 发布 ────────────────────────────────────────────────────

    def _publish_path(self, path_world: List[Tuple[float, float]]) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_stop(self) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
