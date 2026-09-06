import argparse
import time
from collections import OrderedDict, deque

import cv2
import numpy as np

import part3

# The intrinsic matrix supplied for part 4, and the one object of known size.
K_GIVEN = np.array([[2564.3186869, 0.0, 0.0],
                    [0.0, 2569.70273111, 0.0],
                    [0.0, 0.0, 1.0]])
CIRCLE_RADIUS_INCHES = 10.0


class Camera:
    """A pinhole camera: pixels in, directions out, plus a scale from one
    object of known size."""

    def __init__(self, K=K_GIVEN, image_size=None, centre_principal_point=False):
        self.K = np.asarray(K, dtype=float)
        if centre_principal_point:
            if image_size is None:
                raise ValueError("image_size is needed to centre the principal point")
            width, height = image_size
            self.K = self.K.copy()
            self.K[0, 2], self.K[1, 2] = width / 2.0, height / 2.0

    fx = property(lambda self: self.K[0, 0])
    fy = property(lambda self: self.K[1, 1])
    cx = property(lambda self: self.K[0, 2])
    cy = property(lambda self: self.K[1, 2])

    def depth_from_circle(self, area_px, radius=CIRCLE_RADIUS_INCHES):
        """Depth of a circle of known radius, from the area of its image."""
        if area_px <= 0:
            return None
        return radius * np.sqrt(np.pi * self.fx * self.fy / area_px)

    def backproject(self, u, v, depth):
        """Pixel plus depth to a 3D point, in whatever units the radius was."""
        return np.array([(u - self.cx) * depth / self.fx,
                         (v - self.cy) * depth / self.fy,
                         depth])


class DepthFromCircle:
    # Tracks the working depth of the plane the shapes lie on.

    def __init__(self, camera, window=31):
        self.camera = camera
        self.samples = deque(maxlen=window)
        self.depth = None

    def update(self, objects):
        for obj in objects.values():
            if obj.get("merged") or obj["clipped"]:
                continue
            if part3.ShapeTracker.label_of(obj) != "circle":
                continue
            estimate = self.camera.depth_from_circle(cv2.contourArea(obj["contour"]))
            if estimate is not None:
                self.samples.append(estimate)
        if self.samples:
            self.depth = float(np.median(self.samples))
        return self.depth


def annotate_3d(frame, objects, depth, camera, fps=None):
    # Part 3's annotation, plus a 3D coordinate under each centre.
    part3.annotate(frame, objects, fps)
    for oid, obj in objects.items():
        u, v = obj["centroid"]
        color = part3.color_for(oid)
        if depth is None:
            part3.draw_text(frame, "3D: no circle seen yet",
                            (int(u) + 14, int(v) + 30), color, 0.5)
            continue
        x, y, z = camera.backproject(u, v, depth)
        part3.draw_text(frame, f"({x:+.1f}, {y:+.1f}, {z:.1f}) in",
                        (int(u) + 14, int(v) + 30), color, 0.5)
    if depth is not None:
        part3.draw_text(frame, f"plane depth {depth:.1f} in  ({depth/12:.1f} ft)",
                        (20, 80), (0, 255, 0), 0.7)
    return frame


def run(source, output=None, scale=0.5, show=True, limit=None, debug=False,
        centre_principal_point=False, radius=CIRCLE_RADIUS_INCHES):
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"could not open {source}")

    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    camera = Camera(image_size=size, centre_principal_point=centre_principal_point)
    print(f"fx={camera.fx:.2f} fy={camera.fy:.2f} "
          f"principal point=({camera.cx:.1f}, {camera.cy:.1f})  "
          f"circle radius={radius} in")
    if not centre_principal_point:
        print("note: principal point is the image corner, as given. "
              "--centre-principal-point measures X and Y from the view centre.")

    pipeline = part3.Pipeline(scale=scale)
    ranging = DepthFromCircle(camera)
    writer, smoothed_fps, frames = None, None, 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()
            objects, mask = pipeline(frame)
            full = pipeline.scaled_to_frame(objects)       # intrinsics are full-res
            depth = ranging.update(full)
            elapsed = time.perf_counter() - started
            instant = 1.0 / max(elapsed, 1e-6)
            smoothed_fps = instant if smoothed_fps is None \
                else 0.9 * smoothed_fps + 0.1 * instant

            view = annotate_3d(frame, full, depth, camera, smoothed_fps)
            if debug and mask is not None:
                inset = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), None,
                                   fx=0.25 / scale, fy=0.25 / scale)
                view[-inset.shape[0]:, -inset.shape[1]:] = inset

            if output:
                if writer is None:
                    src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps, (view.shape[1], view.shape[0]))
                writer.write(view)
            if show:
                cv2.imshow("part4", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 60 == 0:
                where = "-" if depth is None else f"{depth:6.1f} in"
                print(f"frame {frames:5d} | {len(objects)} tracked "
                      f"| plane depth {where} | {smoothed_fps:.1f} fps")
            if limit and frames >= limit:
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    print(f"done: {frames} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D shape centres from one known circle")
    parser.add_argument("source", help="video path, camera index, or stream URL")
    parser.add_argument("-o", "--output", help="write the annotated video here")
    parser.add_argument("-s", "--scale", type=float, default=0.5,
                        help="detection downscale factor (default 0.5)")
    parser.add_argument("--no-show", action="store_true", help="headless")
    parser.add_argument("--limit", type=int, help="stop after N frames")
    parser.add_argument("--debug", action="store_true", help="inset the bitmask")
    parser.add_argument("--centre-principal-point", action="store_true",
                        help="put the principal point at the image centre "
                             "instead of the (0, 0) the given matrix specifies")
    parser.add_argument("--radius", type=float, default=CIRCLE_RADIUS_INCHES,
                        help="real circle radius in inches (default 10)")
    a = parser.parse_args()
    run(a.source, a.output, a.scale, not a.no_show, a.limit, a.debug,
        a.centre_principal_point, a.radius)
