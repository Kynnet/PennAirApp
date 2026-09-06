"""Run the whole system: video in, shapes out.

    ros2 launch pennair_vision shapes.launch.py video:=/path/to/clip.mp4

Both nodes are started in one process group, so Ctrl-C stops the pair. The
detector reads whatever the publisher puts on 'image_raw', which means the
publisher can be swapped for a real camera driver without touching the
detector -- run any node that publishes sensor_msgs/Image on that topic and
the rest of the system does not notice the difference.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The launch file is installed into share/, so the repository is located by
# environment variable, falling back to the usual clone location.
DEFAULT_REPO = os.environ.get("PENNAIR_REPO", str(Path.home() / "PennAirApp"))


def generate_launch_description():
    video = LaunchConfiguration("video")
    repo = LaunchConfiguration("repo")
    scale = LaunchConfiguration("scale")
    loop = LaunchConfiguration("loop")
    rviz = LaunchConfiguration("rviz")
    view = LaunchConfiguration("view")

    return LaunchDescription([
        DeclareLaunchArgument(
            "video",
            default_value=os.path.join(DEFAULT_REPO, "PennAir_2024_App_Dynamic_Hard.mp4"),
            description="video file to stream"),
        DeclareLaunchArgument(
            "repo", default_value=DEFAULT_REPO,
            description="directory holding part3.py and part4.py"),
        DeclareLaunchArgument(
            "scale", default_value="0.5",
            description="detection downscale factor; lower is faster"),
        DeclareLaunchArgument(
            "loop", default_value="true",
            description="restart the video when it ends"),
        DeclareLaunchArgument(
            "view", default_value="true",
            description="open a window showing the annotated video; set false "
                        "when running over SSH or without a desktop"),
        DeclareLaunchArgument(
            "rviz", default_value="false",
            description="also open RViz"),

        Node(
            package="pennair_vision",
            executable="video_publisher",
            name="video_publisher",
            output="screen",
            parameters=[{
                "video_path": video,
                "loop": loop,
                "frame_id": "camera",
            }],
            remappings=[("image_raw", "/camera/image_raw")],
        ),

        Node(
            package="pennair_vision",
            executable="shape_detector",
            name="shape_detector",
            output="screen",
            parameters=[{
                "algorithm_path": repo,
                "scale": scale,
                "circle_radius_in": 10.0,
                "centre_principal_point": False,
                "publish_annotated": True,
                "publish_markers": True,
            }],
            remappings=[
                ("image_raw", "/camera/image_raw"),
                ("shapes", "/shapes"),
                ("image_annotated", "/shapes/image_annotated"),
                ("markers", "/shapes/markers"),
            ],
        ),

        # Neither node opens a window of its own -- they publish, and a
        # viewer is a separate process. Without this the system runs
        # correctly and appears to do nothing.
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="image_view",
            arguments=["/shapes/image_annotated"],
            output="screen",
            condition=IfCondition(view),
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(rviz),
        ),
    ])
