from setuptools import setup, find_packages

package_name = 'go2_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/navigation.launch.py',
            'launch/test_planning.launch.py',
            'launch/test_planning_nolidar.launch.py',
            'launch/test_avoidance.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/navigation_params.yaml',
            'config/navigation.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='w',
    maintainer_email='dev@example.com',
    description='Go2 自主导航系统',
    license='MIT',
    entry_points={
        'console_scripts': [
            'map_server = go2_navigation.map_server:main',
            'path_planner = go2_navigation.path_planner:main',
            'obstacle_avoider = go2_navigation.obstacle_avoider:main',
            'motion_controller = go2_navigation.motion_controller:main',
            'webrtc_bridge = go2_navigation.webrtc_bridge:main',
            'navigation_node = go2_navigation.navigation_node:main',
            'manual_pose = go2_navigation.manual_pose:main',
            'tilt_monitor = go2_navigation.tilt_monitor:main',
            'odometry_to_base_link = go2_navigation.odometry_to_base_link:main',
        ],
    },
)
