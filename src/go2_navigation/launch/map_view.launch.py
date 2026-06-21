"""Map-only viewer for tuning the offline 3D navigation graph.

This launch starts only map_server and optional RViz. It does not start
path planning, motion control, obstacle avoidance, or WebRTC.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('go2_navigation')
    config_file = os.path.join(pkg_dir, 'config', 'navigation_params.yaml')
    rviz_config = os.path.join(pkg_dir, 'config', 'navigation.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('config', default_value=config_file, description='Parameter file'),
        DeclareLaunchArgument('rviz_config', default_value=rviz_config, description='RViz config'),
        DeclareLaunchArgument('nav_graph_source', default_value='ground_grid', description='ground_grid, free_space, or ground_points'),
        DeclareLaunchArgument('ground_support_radius', default_value='1.0', description='Max distance to real ground points'),
        DeclareLaunchArgument('robot_clearance', default_value='0.0', description='Obstacle inflation radius'),
        DeclareLaunchArgument('free_space_resolution', default_value='0.4', description='Navigation graph sample spacing'),

        Node(
            package='go2_navigation',
            executable='map_server',
            name='map_server',
            parameters=[
                LaunchConfiguration('config'),
                {
                    'nav_graph_source': LaunchConfiguration('nav_graph_source'),
                    'ground_support_radius': LaunchConfiguration('ground_support_radius'),
                    'robot_clearance': LaunchConfiguration('robot_clearance'),
                    'free_space_resolution': LaunchConfiguration('free_space_resolution'),
                },
            ],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', LaunchConfiguration('rviz_config')],
            output='screen',
        ),
    ])
