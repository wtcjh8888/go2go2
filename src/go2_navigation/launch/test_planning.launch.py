"""测试模式 1：纯路径规划 + 跟踪（无避障）。

数据流：
  map_server → /map → path_planner → /planned_path → motion_controller → /cmd_vel → webrtc_bridge → Go2

用法:
    ros2 launch go2_navigation test_planning.launch.py
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
    rviz_config = os.path.join(pkg_dir, 'config', 'navigation.rviz')

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

        # 运动控制器（直接输出到 /cmd_vel，不经过避障）
        Node(
            package='go2_navigation',
            executable='motion_controller',
            name='motion_controller',
            parameters=[{
                'look_ahead_distance': 0.5,
                'control_frequency': 10.0,
                'max_linear_speed': 0.2,
                'max_angular_speed': 0.5,
                'max_acceleration': 0.3,
                'goal_tolerance': 0.15,
                'odom_topic': '/Odometry',
                'cmd_vel_topic': '/cmd_vel',  # 直接输出到 /cmd_vel
            }],
            output='screen',
        ),

        # WebRTC Bridge
        Node(
            package='go2_navigation',
            executable='webrtc_bridge',
            name='webrtc_bridge',
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),

        # RViz（自动加载配置）
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
