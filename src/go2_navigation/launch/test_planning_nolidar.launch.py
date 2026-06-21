"""测试模式 1b：纯路径规划（无 LiDAR，手动设置位置）。

不需要 LiDAR 和 FAST-LIO。
在 RViz 中用 "2D Pose Estimate" 设置机器人位置，用 "2D Nav Goal" 设置目标。

用法:
    ros2 launch go2_navigation test_planning_nolidar.launch.py
    # 或指定初始位置
    ros2 launch go2_navigation test_planning_nolidar.launch.py x:=1.0 y:=2.0
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
        DeclareLaunchArgument('x', default_value='0.0', description='初始 X 坐标'),
        DeclareLaunchArgument('y', default_value='0.0', description='初始 Y 坐标'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='初始朝向 (弧度)'),

        # 手动位姿（替代 FAST-LIO 的 /Odometry）
        Node(
            package='go2_navigation',
            executable='manual_pose',
            name='manual_pose',
            parameters=[{
                'x': LaunchConfiguration('x'),
                'y': LaunchConfiguration('y'),
                'yaw': LaunchConfiguration('yaw'),
                'odom_topic': '/Odometry',
                'initial_pose_topic': '/initial_pose',
            }],
            output='screen',
        ),

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

        # 运动控制器
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
                'cmd_vel_topic': '/cmd_vel',
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
