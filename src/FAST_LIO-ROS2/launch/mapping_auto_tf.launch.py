
import math

import os

import time



from ament_index_python.packages import get_package_share_directory



from launch import LaunchDescription

from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction

from launch.conditions import IfCondition

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node





def _bool_value(value):

    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')





def _sample_imu_average(imu_topic, sample_duration, sample_timeout, min_samples):

    import rclpy

    from sensor_msgs.msg import Imu



    samples = []

    rclpy.init(args=None)

    node = rclpy.create_node('fast_lio_auto_base_link_calibrator')



    def cb(msg):

        a = msg.linear_acceleration

        samples.append((float(a.x), float(a.y), float(a.z)))



    sub = node.create_subscription(Imu, imu_topic, cb, 200)

    # Keep subscription alive while spinning.



    start = None

    deadline = time.monotonic() + sample_timeout

    while rclpy.ok() and time.monotonic() < deadline:

        rclpy.spin_once(node, timeout_sec=0.05)

        if samples and start is None:

            start = time.monotonic()

        if start is not None and time.monotonic() - start >= sample_duration and len(samples) >= min_samples:

            break



    node.destroy_node()

    rclpy.shutdown()



    if len(samples) < min_samples:

        raise RuntimeError(

            f'Only received {len(samples)} IMU samples from {imu_topic}; '

            f'need at least {min_samples}. Start Livox driver first and keep the robot still.'

        )



    ax = sum(s[0] for s in samples) / len(samples)

    ay = sum(s[1] for s in samples) / len(samples)

    az = sum(s[2] for s in samples) / len(samples)

    norm = math.sqrt(ax * ax + ay * ay + az * az)

    if norm < 1e-9:

        raise RuntimeError('IMU acceleration norm is too small; cannot estimate gravity direction.')



    # Livox IMU acceleration is opposite to FAST-LIO's printed gravity vector.

    # For body -> base_link compensation:

    #   positive body Y gravity component needs positive roll compensation;

    #   positive body X gravity component needs pitch compensation with the same sign as ax.

    roll = -math.asin(max(-1.0, min(1.0, ay / norm)))

    pitch = math.asin(max(-1.0, min(1.0, ax / norm)))

    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(az) / norm))))

    return {

        'count': len(samples),

        'ax': ax,

        'ay': ay,

        'az': az,

        'norm': norm,

        'roll': roll,

        'pitch': pitch,

        'tilt': tilt,

    }





def _launch_after_calibration(context, *args, **kwargs):

    package_path = get_package_share_directory('fast_lio')

    default_config_path = os.path.join(package_path, 'config')

    default_rviz_config_path = os.path.join(package_path, 'rviz', 'fastlio.rviz')



    use_sim_time = _bool_value(LaunchConfiguration('use_sim_time').perform(context))

    config_path = LaunchConfiguration('config_path').perform(context) or default_config_path

    config_file = LaunchConfiguration('config_file').perform(context)

    rviz_use = LaunchConfiguration('rviz').perform(context)

    rviz_cfg = LaunchConfiguration('rviz_cfg').perform(context) or default_rviz_config_path

    imu_topic = LaunchConfiguration('imu_topic').perform(context)

    sample_duration = float(LaunchConfiguration('sample_duration').perform(context))

    sample_timeout = float(LaunchConfiguration('sample_timeout').perform(context))

    min_samples = int(LaunchConfiguration('min_samples').perform(context))

    base_x = LaunchConfiguration('base_x').perform(context)

    base_y = LaunchConfiguration('base_y').perform(context)

    base_z = LaunchConfiguration('base_z').perform(context)

    yaw = float(LaunchConfiguration('yaw').perform(context))

    auto_roll = _bool_value(LaunchConfiguration('auto_roll').perform(context))

    auto_pitch = _bool_value(LaunchConfiguration('auto_pitch').perform(context))



    result = _sample_imu_average(imu_topic, sample_duration, sample_timeout, min_samples)

    roll = result['roll'] if auto_roll else 0.0

    pitch = result['pitch'] if auto_pitch else 0.0



    summary = (

        '[auto_tf] IMU samples={count}, acc=[{ax:.5f}, {ay:.5f}, {az:.5f}], '

        'norm={norm:.5f}, raw_tilt={tilt:.2f} deg, body_to_base roll={roll_deg:.2f} deg, '

        'pitch={pitch_deg:.2f} deg'

    ).format(

        count=result['count'],

        ax=result['ax'],

        ay=result['ay'],

        az=result['az'],

        norm=result['norm'],

        tilt=result['tilt'],

        roll_deg=math.degrees(roll),

        pitch_deg=math.degrees(pitch),

    )



    fast_lio_node = Node(

        package='fast_lio',

        executable='fastlio_mapping',

        parameters=[PathJoinSubstitution([config_path, config_file]), {'use_sim_time': use_sim_time}],

        output='screen',

    )



    base_link_tf = Node(

        package='tf2_ros',

        executable='static_transform_publisher',

        arguments=[

            '--x', base_x,

            '--y', base_y,

            '--z', base_z,

            '--roll', f'{roll:.8f}',

            '--pitch', f'{pitch:.8f}',

            '--yaw', f'{yaw:.8f}',

            '--frame-id', 'body',

            '--child-frame-id', 'base_link',

        ],

    )



    rviz_node = Node(

        package='rviz2',

        executable='rviz2',

        arguments=['-d', rviz_cfg],

        condition=IfCondition(rviz_use),

    )



    return [LogInfo(msg=summary), fast_lio_node, base_link_tf, rviz_node]





def generate_launch_description():

    package_path = get_package_share_directory('fast_lio')

    default_config_path = os.path.join(package_path, 'config')

    default_rviz_config_path = os.path.join(package_path, 'rviz', 'fastlio.rviz')



    return LaunchDescription([

        DeclareLaunchArgument('use_sim_time', default_value='false'),

        DeclareLaunchArgument('config_path', default_value=default_config_path),

        DeclareLaunchArgument('config_file', default_value='mid360.yaml'),

        DeclareLaunchArgument('rviz', default_value='true'),

        DeclareLaunchArgument('rviz_cfg', default_value=default_rviz_config_path),

        DeclareLaunchArgument('imu_topic', default_value='/livox/imu'),

        DeclareLaunchArgument('sample_duration', default_value='2.0'),

        DeclareLaunchArgument('sample_timeout', default_value='8.0'),

        DeclareLaunchArgument('min_samples', default_value='100'),

        DeclareLaunchArgument('base_x', default_value='0.20'),

        DeclareLaunchArgument('base_y', default_value='0'),

        DeclareLaunchArgument('base_z', default_value='0.05'),

        DeclareLaunchArgument('yaw', default_value='0'),

        DeclareLaunchArgument('auto_roll', default_value='true'),

        DeclareLaunchArgument('auto_pitch', default_value='true'),

        OpaqueFunction(function=_launch_after_calibration),

    ])

