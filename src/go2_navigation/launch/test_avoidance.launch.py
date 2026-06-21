"""测试模式 2：路径规划 + 实时避障。

数据流：
  map_server → /map → path_planner → /planned_path → motion_controller
  motion_controller → /cmd_vel_raw → obstacle_avoider → /cmd_vel → webrtc_bridge → Go2

用法:
    ros2 launch go2_navigation test_avoidance.launch.py
    # 然后在 RViz 中用 "2D Nav Goal" 点击目标点
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('go2_navigation')
    config_file = os.path.join(pkg_dir, 'config', 'navigation_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config_file),

        # 地图服务器
        Node(
            package='go2_navigation',
            executable='map_server',
            name='map_server',
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),

        # 路径规划器
        Node(
            package='go2_navigation',
            executable='path_planner',
            name='path_planner',
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),

        # 运动控制器（输出到 /cmd_vel_raw，给避障模块处理）
        Node(
            package='go2_navigation',
            executable='motion_controller',
            name='motion_controller',
            parameters=[{
                'look_ahead_distance': 0.3,
                'control_frequency': 10.0,
                'max_linear_speed': 0.2,
                'max_angular_speed': 0.5,
                'max_acceleration': 0.3,
                'goal_tolerance': 0.15,
                'odom_topic': '/Odometry',
                'cmd_vel_topic': '/cmd_vel_raw',  # 输出给避障模块
            }],
            output='screen',
        ),

        # 避障模块
        Node(
            package='go2_navigation',
            executable='obstacle_avoider',
            name='obstacle_avoider',
            parameters=[{
                'cloud_topic': '/cloud_registered_body',
                'detection_range': 2.0,
                'danger_distance': 0.3,
                'warning_distance': 0.6,
                'fov_angle': 120.0,
                'max_linear_speed': 0.2,
                'max_angular_speed': 0.5,
                'input_cmd_vel_topic': '/cmd_vel_raw',
                'output_cmd_vel_topic': '/cmd_vel',
            }],
            output='screen',
        ),

        # WebRTC Bridge
        Node(
            package='go2_navigation',
            executable='webrtc_bridge',
            name='webrtc_bridge',
            parameters=[{
                'robot_ip': '192.168.12.1',
                'aes_key': '7b7ad05fae7b79f3c0135f7417f895d0',
                'cmd_vel_topic': '/cmd_vel',
                'max_linear_vel': 0.2,
                'max_angular_vel': 0.5,
            }],
            output='screen',
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
