from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py'
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            # Enable depth/color alignment -- required for this pipeline.
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'false',  # we build our own masked clouds
            # Matches the calibrated intrinsics in pointcloud_utils.py
            # (fx/fy/cx/cy are resolution-specific -- keep these in sync).
            'rgb_camera.color_profile': '848x480x30',
            'depth_module.depth_profile': '848x480x30',
        }.items(),
    )

    segmentation_node = Node(
        package='ycb_pointcloud_pipeline',
        executable='segmentation_node',
        name='ycb_segmentation_node',
        output='screen',
        parameters=[{
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'detector_mode': 'grounded_sam',  # 'dummy' | 'sam_all' | 'grounded_sam'
            'save_pcd': True,
            'show_debug_window': True,
            'detect_every_n_frames': 1,  # Jetson Thor has a GPU -- raise this only if it lags
            # 'target_objects_yaml': '/custom/path.yaml',  # defaults to config/target_objects.yaml
        }],
    )

    return LaunchDescription([realsense_launch, segmentation_node])
