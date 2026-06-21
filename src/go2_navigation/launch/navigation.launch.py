"""Go2 导航系统启动文件。

启动全部导航节点 + RViz 可视化。

用法:
    ros2 launch go2_navigation navigation.launch.py
    ros2 launch go2_navigation navigation.launch.py rviz:=false  # 不启动 RViz
    ros2 launch go2_navigation navigation.launch.py mode:=integrated  # 一体化模式（navigation_node）
    ros2 launch go2_navigation navigation.launch.py mode:=modular     # 分模块模式（独立节点）
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('go2_navigation')
    config_file = os.path.join(pkg_dir, 'config', 'navigation_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'navigation.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='启动 RViz'),
        DeclareLaunchArgument('config', default_value=config_file, description='参数配置文件'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config, description='RViz 配置文件'),
        DeclareLaunchArgument('mode', default_value='modular', description='运行模式: modular(分模块) 或 integrated(一体化)'),

        Node(
            package='go2_navigation',
            executable='odometry_to_base_link',
            name='odometry_to_base_link',
            parameters=[LaunchConfiguration('config')],
            output='screen',
        ),

        # ── 分模块模式：每个模块独立运行 ──
        GroupAction([
            # 模块 1: 地图服务器
            Node(
                package='go2_navigation',
                executable='map_server',
                name='map_server',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # 模块 2: 路径规划器
            Node(
                package='go2_navigation',
                executable='path_planner',
                name='path_planner',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # 模块 3: 实时避障
            Node(
                package='go2_navigation',
                executable='obstacle_avoider',
                name='obstacle_avoider',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # 模块 4: 运动控制器
            Node(
                package='go2_navigation',
                executable='motion_controller',
                name='motion_controller',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # 模块 5: WebRTC Bridge
            Node(
                package='go2_navigation',
                executable='webrtc_bridge',
                name='webrtc_bridge',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),
        ], condition=UnlessCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'integrated'"]))),

        # ── 一体化模式：navigation_node 包含所有功能 ──
        GroupAction([
            # 地图服务器（仍需独立运行）
            Node(
                package='go2_navigation',
                executable='map_server',
                name='map_server',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # 一体化导航节点
            Node(
                package='go2_navigation',
                executable='navigation_node',
                name='navigation_node',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),

            # WebRTC Bridge（仍需独立运行）
            Node(
                package='go2_navigation',
                executable='webrtc_bridge',
                name='webrtc_bridge',
                parameters=[LaunchConfiguration('config')],
                output='screen',
            ),
        ], condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'integrated'"]))),

        # ── RViz（默认配置，手动添加话题）──
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
