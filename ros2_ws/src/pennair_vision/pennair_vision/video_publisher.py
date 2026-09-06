"""Publish a video file as a stream of ROS images.

This is the first of the two nodes: it does no vision work at all, it only
turns a file into the same thing a real camera driver would produce, so that
everything downstream is written against a live camera and does not know or
care that the frames came from disk.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("loop", True)
        self.declare_parameter("frame_rate", 0.0)   # 0 => use the file's own rate
        self.declare_parameter("frame_id", "camera")

        path = self.get_parameter("video_path").value
        if not path:
            raise SystemExit("video_path parameter is required")

        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise SystemExit(f"could not open {path}")

        self.loop = self.get_parameter("loop").value
        self.frame_id = self.get_parameter("frame_id").value
        rate = self.get_parameter("frame_rate").value or \
            self.capture.get(cv2.CAP_PROP_FPS) or 30.0

        self.bridge = CvBridge()
        # Sensor QoS: best effort, keep last. A frame is only useful while it
        # is current, so dropping one under load beats queueing it and falling
        # further behind. The detector subscribes with the same profile --
        # a reliable subscriber will not connect to a best-effort publisher.
        self.publisher = self.create_publisher(Image, "image_raw", qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.frames = 0
        self.get_logger().info(f"publishing {path} on 'image_raw' at {rate:.2f} fps")

    def tick(self):
        ok, frame = self.capture.read()
        if not ok:
            if not self.loop:
                self.get_logger().info("end of video, shutting down")
                raise SystemExit(0)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                raise SystemExit("could not rewind the video")

        message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)

        self.frames += 1
        if self.frames % 120 == 0:
            self.get_logger().info(f"published {self.frames} frames")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
