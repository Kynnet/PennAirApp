# ROS 2 workspace

Two nodes and a launch file. `video_publisher` turns a video file into a
`sensor_msgs/Image` stream; `shape_detector` subscribes to that stream, runs
the part 3 detector and the part 4 projection, and publishes what it found.

The split matters: the detector never learns where its images came from, so
swapping the file publisher for a real camera driver changes nothing
downstream as long as the driver publishes `sensor_msgs/Image`.

## Topics

| topic | type | contents |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/Image` | the raw video frames |
| `/shapes` | `pennair_msgs/ShapeArray` | id, name, edge count, outline, centre in pixels, position in inches |
| `/shapes/image_annotated` | `sensor_msgs/Image` | the drawn-on frame |
| `/shapes/markers` | `visualization_msgs/MarkerArray` | outlines and labels for RViz |

## One-time setup on the VM

ROS's Python is the system Python, not a virtualenv, so the dependencies go
in with apt:

    sudo apt update
    sudo apt install -y python3-opencv python3-scipy python3-numpy \
                        ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv \
                        ros-$ROS_DISTRO-rviz2 ros-$ROS_DISTRO-rqt-image-view

Tell the nodes where the repository is, so they can import `part3.py` and
`part4.py` rather than keeping a second copy of them:

    echo 'export PENNAIR_REPO=~/PennAirApp' >> ~/.bashrc
    source ~/.bashrc

## Build

    cd ~/PennAirApp/ros2_ws
    source /opt/ros/$ROS_DISTRO/setup.bash
    colcon build
    source install/setup.bash

`pennair_msgs` must build before `pennair_vision` can import it; colcon works
that out from the dependency in `package.xml`, so plain `colcon build` is
enough. Re-source `install/setup.bash` in every new terminal.

## Run

    ros2 launch pennair_vision shapes.launch.py

or with arguments:

    ros2 launch pennair_vision shapes.launch.py \
        video:=$PENNAIR_REPO/PennAir_2024_App_Dynamic.mp4 scale:=0.4 rviz:=true

## Check it is working

    ros2 topic list
    ros2 topic hz /camera/image_raw
    ros2 topic echo /shapes --once
    ros2 run rqt_image_view rqt_image_view /shapes/image_annotated

In RViz set the fixed frame to `camera` and add a MarkerArray display on
`/shapes/markers`. The markers are in pixel coordinates, so the view will be
about 1920 by 1080 units across.

## Notes

`scale` is the detection downscale factor. A VM has no GPU and limited cores,
so if the detector cannot keep up, lower it (`scale:=0.4` or `0.35`) before
anything else. Frames are dropped rather than queued -- both image topics use
sensor QoS, which is best-effort -- so falling behind costs frames, not
latency. `ros2 topic hz /shapes` against `ros2 topic hz /camera/image_raw`
shows how many are getting through.
