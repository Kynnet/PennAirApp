"""Detect shapes in an incoming image stream and publish them.

This is the second node. It subscribes to the images the first one puts out,
runs part 3's detector and part 4's projection on each frame, and publishes
what it found.

The detection code itself is not duplicated here. part3.py and part4.py stay
the single source of truth at the top of the repository, and this node adds
that directory to sys.path and imports them. Copying them into the package
would be simpler to build and would guarantee the two copies drift apart the
first time either is edited.

Three topics come out:

  shapes           pennair_msgs/ShapeArray -- ids, names, outlines, centres
                   in pixels and, once the circle has set the scale, in
                   inches. This is the machine-readable one.
  image_annotated  sensor_msgs/Image -- the same picture the desktop version
                   draws, for looking at in rqt_image_view.
  markers          visualization_msgs/MarkerArray -- outlines and labels for
                   RViz, so the results can be seen without custom plugins.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pennair_msgs.msg import Shape, ShapeArray


def load_algorithms(hint):
    """Import part3/part4 from the repository, wherever it has been put."""
    candidates = [hint, os.environ.get("PENNAIR_REPO", ""),
                  str(Path.home() / "PennAirApp"), os.getcwd()]
    for candidate in candidates:
        if candidate and (Path(candidate) / "part3.py").is_file():
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            import part3
            import part4
            return part3, part4, candidate
    raise SystemExit(
        "could not find part3.py. Set the algorithm_path parameter or the "
        f"PENNAIR_REPO environment variable. Looked in: {candidates}")


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector")

        self.declare_parameter("algorithm_path", "")
        self.declare_parameter("scale", 0.5)
        self.declare_parameter("circle_radius_in", 10.0)
        self.declare_parameter("centre_principal_point", False)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_markers", True)
        # Width the intrinsic matrix was measured at. If images arrive
        # smaller than this, K is scaled to match; the 3D output is
        # unchanged either way.
        self.declare_parameter("intrinsics_width", 1920)
        # Writing the annotated video to a file is the headless alternative
        # to a viewer window: over SSH there is no display to draw on, and
        # the file can be copied back and watched anywhere.
        self.declare_parameter("record_path", "")
        self.declare_parameter("record_fps", 15.0)

        self.part3, self.part4, where = load_algorithms(
            self.get_parameter("algorithm_path").value)
        self.get_logger().info(f"loaded detection algorithms from {where}")

        self.radius = self.get_parameter("circle_radius_in").value
        self.publish_annotated = self.get_parameter("publish_annotated").value
        self.publish_markers = self.get_parameter("publish_markers").value
        self.pipeline = self.part3.Pipeline(scale=self.get_parameter("scale").value)

        # The camera model needs the image size, which is only known once a
        # frame has arrived, so it is built lazily in the callback.
        self.camera = None
        self.ranging = None
        self.centre_pp = self.get_parameter("centre_principal_point").value

        self.record_path = self.get_parameter("record_path").value
        self.record_fps = float(self.get_parameter("record_fps").value)
        self.writer = None

        self.bridge = CvBridge()
        self.shapes_pub = self.create_publisher(ShapeArray, "shapes", 10)
        self.image_pub = self.create_publisher(Image, "image_annotated", qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, "markers", 10)
        self.subscription = self.create_subscription(
            Image, "image_raw", self.on_image, qos_profile_sensor_data)

        self.frames = 0
        self.reported_at = time.monotonic()
        self.reported_frames = 0
        self.get_logger().info("waiting for images on 'image_raw'")

    # -- per frame ---------------------------------------------------------

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

        if self.camera is None:
            height, width = frame.shape[:2]
            native = int(self.get_parameter("intrinsics_width").value)
            factor = width / float(native)
            K = self.part4.K_GIVEN.copy()
            if abs(factor - 1.0) > 1e-6:
                K[0, 0] *= factor      # fx
                K[1, 1] *= factor      # fy
                K[0, 2] *= factor      # cx
                K[1, 2] *= factor      # cy
                self.get_logger().info(
                    f"images arrive at {width}px wide against intrinsics for "
                    f"{native}px; scaling K by {factor:.3f}")
            self.camera = self.part4.Camera(
                K=K, image_size=(width, height),
                centre_principal_point=self.centre_pp)
            self.ranging = self.part4.DepthFromCircle(self.camera)
            self.get_logger().info(
                f"camera fx={self.camera.fx:.1f} fy={self.camera.fy:.1f} "
                f"principal point=({self.camera.cx:.1f}, {self.camera.cy:.1f})")

        objects, _ = self.pipeline(frame)
        full = self.pipeline.scaled_to_frame(objects)   # intrinsics are full-res
        depth = self.ranging.update(full)

        self.shapes_pub.publish(self.to_message(full, depth, message.header))
        if self.publish_annotated or self.record_path:
            annotated = self.part4.annotate_3d(frame, full, depth, self.camera)
            if self.publish_annotated:
                out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out.header = message.header
                self.image_pub.publish(out)
            if self.record_path:
                self.record(annotated)
        if self.publish_markers:
            self.marker_pub.publish(self.to_markers(full, message.header))

        self.frames += 1
        now = time.monotonic()
        if now - self.reported_at >= 5.0:
            achieved = (self.frames - self.reported_frames) / (now - self.reported_at)
            where = "unknown" if depth is None else f"{depth:.1f} in"
            self.get_logger().info(
                f"{self.frames} frames | {len(full)} shapes | "
                f"plane depth {where} | {achieved:.2f} fps")
            self.reported_at, self.reported_frames = now, self.frames

    def record(self, annotated):
        if self.writer is None:
            height, width = annotated.shape[:2]
            self.writer = cv2.VideoWriter(
                self.record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                self.record_fps, (width, height))
            if not self.writer.isOpened():
                self.get_logger().error(f"could not open {self.record_path} for writing")
                self.record_path = ""
                self.writer = None
                return
            self.get_logger().info(f"recording annotated video to {self.record_path}")
        self.writer.write(annotated)

    def destroy_node(self):
        # Without releasing it the file has no index and will not play.
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"wrote {self.record_path}")
            self.writer = None
        super().destroy_node()

    # -- conversions -------------------------------------------------------

    def outline_of(self, obj, label, merged):
        if merged:
            return self.part3.smooth(obj["contour"])
        return self.part3.outline_of(obj["contour"], label)

    def to_message(self, objects, depth, header):
        array = ShapeArray()
        array.header = header
        array.plane_depth = float(depth) if depth is not None else 0.0

        for oid, obj in objects.items():
            merged = bool(obj.get("merged", False))
            label = "merged shapes" if merged else self.part3.ShapeTracker.label_of(obj)
            polygon = self.outline_of(obj, label, merged).reshape(-1, 2)

            shape = Shape()
            shape.id = int(oid)
            shape.label = label
            shape.edges = 0 if label == "circle" else int(len(polygon))
            shape.overlapped = bool(obj["overlapped"])
            shape.clipped = bool(obj["clipped"])

            u, v = obj["centroid"]
            shape.centroid = Point(x=float(u), y=float(v), z=0.0)
            shape.outline = [Point(x=float(px), y=float(py), z=0.0)
                             for px, py in polygon]

            if depth is not None:
                x, y, z = self.camera.backproject(u, v, depth)
                shape.position = Point(x=float(x), y=float(y), z=float(z))
                shape.has_position = True
            else:
                shape.has_position = False

            array.shapes.append(shape)
        return array

    def to_markers(self, objects, header):
        markers = MarkerArray()
        wipe = Marker()
        wipe.header = header
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)

        for oid, obj in objects.items():
            merged = bool(obj.get("merged", False))
            label = "merged shapes" if merged else self.part3.ShapeTracker.label_of(obj)
            polygon = self.outline_of(obj, label, merged).reshape(-1, 2)
            blue, green, red = self.part3.color_for(oid)      # part 3 stores BGR

            line = Marker()
            line.header = header
            line.ns = "outline"
            line.id = int(oid)
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 2.0
            line.color = ColorRGBA(r=red / 255.0, g=green / 255.0,
                                   b=blue / 255.0, a=1.0)
            line.points = [Point(x=float(px), y=float(py), z=0.0)
                           for px, py in polygon]
            if line.points:
                line.points.append(line.points[0])            # close the loop
            markers.markers.append(line)

            text = Marker()
            text.header = header
            text.ns = "label"
            text.id = int(oid)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = 24.0
            text.color = line.color
            u, v = obj["centroid"]
            text.pose.position = Point(x=float(u), y=float(v) - 30.0, z=0.0)
            text.pose.orientation.w = 1.0
            text.text = f"#{oid} {label}"
            markers.markers.append(text)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ShapeDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"[shape_detector] {exc.code}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
