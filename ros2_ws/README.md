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
| `/shapes/markers` | `visualization_msgs/MarkerArray` | outlines and labels for RViz |

## One-time setup on the VM

ROS's Python is the system Python, not a virtualenv, so the dependencies go
in with apt:

    sudo apt update
    sudo apt install -y python3-opencv python3-scipy python3-numpy \
                        ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv \
                        ros-$ROS_DISTRO-rviz2

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

On a VM this is the combination worth starting from:

    ros2 launch pennair_vision shapes.launch.py \
        publish_scale:=0.5 scale:=1.0 rate:=15 loop:=false

### Arguments

| argument | default | meaning |
| --- | --- | --- |
| `video` | the hard clip | video file to stream |
| `repo` | `$PENNAIR_REPO` | directory holding `part3.py` and `part4.py` |
| `publish_scale` | `1.0` | downscale frames before publishing them |
| `scale` | `0.5` | detection downscale, applied inside the detector |
| `rate` | `0.0` | frames per second to publish; 0 means the video's own rate |
| `loop` | `true` | restart the video when it ends |
| `rviz` | `false` | also open RViz |

## Seeing the output

The nodes publish data, not pictures. `/shapes` is the output the system
exists to produce, and it needs no display:

    ros2 topic echo /shapes --once

That prints every shape with its id, name, edge count, outline, centre in
pixels and position in inches, plus the recovered depth of the plane.

For a picture, RViz will draw the outlines: set the fixed frame to `camera`
and add a MarkerArray display on `/shapes/markers`. The markers are in pixel
coordinates, so the view is about 1920 by 1080 units across.

If what you want is an annotated video, run the detector directly rather than
through ROS -- it draws the same overlay and writes a file:

    python3 part4.py PennAir_2024_App_Dynamic_Hard.mp4 -o annotated.mp4 --no-show

The ROS nodes deliberately do not publish or record video. A drawn-on frame
is another 6.2 MB per frame on the wire, which costs more than the detection
does, and it duplicates something the standalone programs already do well.

## Check it is working

    ros2 topic list
    ros2 topic hz /camera/image_raw
    ros2 topic hz /shapes
