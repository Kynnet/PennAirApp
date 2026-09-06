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
from launch_ros.parameter_descriptions import ParameterValue

# The launch file is installed into share/, so the repository is located by
# environment variable, falling back to the usual clone location.
DEFAULT_REPO = os.environ.get("PENNAIR_REPO", str(Path.home() / "PennAirApp"))


def number(configuration):
    """A launch argument spelled as a parameter of a definite type.

    Launch arguments arrive as strings and rclpy infers the type, so
    'rate:=15' becomes an INTEGER and is refused by a parameter declared as a
    DOUBLE, while 'rate:=15.0' is accepted. Stating the type here means both
    spellings work.
    """
    return ParameterValue(configuration, value_type=float)


def flag(configuration):
    return ParameterValue(configuration, value_type=bool)


def generate_launch_description():
    video = LaunchConfiguration("video")
    repo = LaunchConfiguration("repo")
    scale = LaunchConfiguration("scale")
    loop = LaunchConfiguration("loop")
    rviz = LaunchConfiguration("rviz")
    rate = LaunchConfiguration("rate")
    publish_scale = LaunchConfiguration("publish_scale")

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
            "loop", default_value="false",
            description="restart the video when it ends"),
        DeclareLaunchArgument(
            "rate", default_value="0.0",
            description="frames per second to publish; 0 means the video's own "
                        "rate. On a slow machine set this to what the detector "
                        "can actually keep up with, so it sees every frame in "
                        "order instead of dropping most of them"),
        DeclareLaunchArgument(
            "publish_scale", default_value="1.0",
            description="downscale frames before publishing them. 0.5 cuts "
                        "image traffic fourfold and does not change the 3D "
                        "result; pair it with scale:=1.0"),
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
                "loop": flag(loop),
                "frame_rate": number(rate),
                "publish_scale": number(publish_scale),
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
                "scale": number(scale),
                "circle_radius_in": 10.0,
                "centre_principal_point": False,
                "publish_markers": True,
            }],
            remappings=[
                ("image_raw", "/camera/image_raw"),
                ("shapes", "/shapes"),
                ("markers", "/shapes/markers"),
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(rviz),
        ),
    ])
