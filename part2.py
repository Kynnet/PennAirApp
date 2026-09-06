import argparse
import time
from collections import Counter, deque, OrderedDict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def local_std(gray, k):
    #Standard deviation of intensity in a k x k window around each pixel.
    #Var(X) = E[X^2] - E[X]^2
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))


def auto_window(height):
    return max(5, int(round(height / 50)) | 1)  # | 1 forces odd


def segment(img, k=None, min_area_frac=0.003):
    h, w = img.shape[:2]
    k = k or auto_window(h)

    #Return external contours of flat-textured regions.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = local_std(gray, k).astype(np.uint8)

    # Otsu adapts the flat/textured split per frame, which matters on video
    # where lighting drifts as the camera pans.
    _, mask = cv2.threshold(std, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use open and close to remove noise in foreground + background
    # high-variance pixels that the window straddles at each shape's edge.
    open_k = max(3, k) | 1
    close_k = max(3, k // 2) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h * w
    return [c for c in contours if cv2.contourArea(c) > min_area], mask


def split_by_color(lab, contour, bin_size=24, min_area=600, uniform_tol=8):
    x, y, w, h = cv2.boundingRect(contour)
    roi = lab[y:y + h, x:x + w]
    shifted = contour - [x, y]

    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [shifted], -1, 255, -1)

    # Erode before measuring so anti-aliased edge pixels don't inflate the
    # spread and trigger a split on a perfectly ordinary single shape.
    interior = roi[cv2.erode(filled, np.ones((5, 5), np.uint8)) > 0]
    if interior.size == 0 or interior.std(axis=0)[1:].max() < uniform_tol:
        return [contour]

    # Quantise LAB into coarse bins and treat each occupied bin as one shape.
    # This needs no guess at how many shapes are in the blob, unlike k-means.
    q = (roi // bin_size).astype(np.int32)
    key = q[:, :, 0] * 10000 + q[:, :, 1] * 100 + q[:, :, 2]

    parts = []
    for value in np.unique(key[filled > 0]):
        #find pixels that are same
        sub = ((key == value) & (filled > 0)).astype(np.uint8) * 255
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        found, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts += [c + [x, y] for c in found if cv2.contourArea(c) > min_area]

    return parts if len(parts) > 1 else [contour]


#detect shapes in the frame
def detect(frame, scale=0.5, min_area_frac=0.003, min_solidity=0.85):
    work = frame
    if scale != 1.0:
        work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    contours, _ = segment(work, min_area_frac=min_area_frac)
    if not contours:
        return []

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)   # converted once per frame
    h, w = work.shape[:2]
    min_area = min_area_frac * h * w

    keep = []
    for blob in contours:
        for part in split_by_color(lab, blob):
            area = cv2.contourArea(part)

            
            limit = min_area * 0.4 if touches_border(part, work.shape) else min_area
            if area <= limit:
                continue

            # Solidity = area / convex-hull area. Now that overlaps have been
            # split apart, every real shape is convex and scores 0.85+
            hull = cv2.contourArea(cv2.convexHull(part))
            if hull > 0 and area / hull < min_solidity:
                continue
            keep.append(part)

    if scale == 1.0:
        return keep
    
    return [np.round(c / scale).astype(np.int32) for c in keep]


def classify(contour):
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim <= 0:
        return "?"
    if 4 * np.pi * area / (perim * perim) > 0.85:
        return "circle"
    verts = len(cv2.approxPolyDP(contour, 0.04 * perim, True))
    return {3: "triangle", 4: "quadrilateral", 5: "pentagon",
            6: "hexagon"}.get(verts, f"{verts}-gon")


def touches_border(contour, shape, pad=3):
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= pad or y <= pad or x + cw >= w - pad or y + ch >= h - pad


# ----------------------------------------------------------------- tracking


def mean_color(frame, contour):
    """Mean LAB color inside a contour, computed on its bounding box only."""
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return np.zeros(3)
    roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2LAB)
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [contour - [x, y]], -1, 255, -1)
    eroded = cv2.erode(m, np.ones((5, 5), np.uint8))
    if cv2.countNonZero(eroded) > 20:
        m = eroded
    if cv2.countNonZero(m) == 0:
        return np.zeros(3)
    return np.array(cv2.mean(roi, mask=m)[:3])


class CentroidTracker:
    """Tracker giving each shape a stable ID across frames.

    Association uses BOTH position and appearance.

    Matching is solved globally rather than greedily.
    """

    def __init__(self, max_distance=200, max_missing=45, vote_window=9,
                 min_hits=3, max_color_distance=60, color_weight=1.5):
        self.next_id = 0
        self.objects = OrderedDict()   # id -> dict(centroid, contour, ...)
        self.missing = OrderedDict()   # id -> consecutive frames unseen
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.vote_window = vote_window
        # A track must be seen this many times before it's reported. Real
        # shapes persist; a patch of grass that happens to look flat and
        # convex in one frame does not.
        self.min_hits = min_hits
        self.max_color_distance = max_color_distance
        self.color_weight = color_weight

    def _register(self, centroid, contour, label, clipped, color):
        self.objects[self.next_id] = {
            "centroid": centroid, "contour": contour, "clipped": clipped,
            "history": deque([label], maxlen=self.vote_window), "hits": 1,
            "color": color,
        }
        self.missing[self.next_id] = 0
        self.next_id += 1

    def confirmed(self):
        """Tracks that are trustworthy AND currently visible.

        A track that wasn't matched this frame is kept alive internally for
        max_missing frames so it can reclaim its ID if the shape reappears,
        but it must not be reported -- its contour is stale, and drawing it
        would paint a shape onto empty grass.
        """
        return OrderedDict((i, o) for i, o in self.objects.items()
                           if o["hits"] >= self.min_hits and self.missing[i] == 0)

    def _age(self, oid):
        self.missing[oid] += 1
        if self.missing[oid] > self.max_missing:
            del self.objects[oid], self.missing[oid]

    def update(self, contours, frame):
        frame_shape = frame.shape
        detections = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            detections.append({
                "centroid": centroid, "contour": c, "label": classify(c),
                "clipped": touches_border(c, frame_shape),
                "color": mean_color(frame, c),
            })

        if not detections:
            for oid in list(self.missing):
                self._age(oid)
            return self.objects

        if not self.objects:
            for d in detections:
                self._register(d["centroid"], d["contour"], d["label"],
                               d["clipped"], d["color"])
            return self.objects

        ids = list(self.objects.keys())
        old_pos = np.array([self.objects[i]["centroid"] for i in ids], float)
        new_pos = np.array([d["centroid"] for d in detections], float)
        old_col = np.array([self.objects[i]["color"] for i in ids], float)
        new_col = np.array([d["color"] for d in detections], float)

        spatial = np.linalg.norm(old_pos[:, None] - new_pos[None, :], axis=2)
        color = np.linalg.norm(old_col[:, None] - new_col[None, :], axis=2)

        # Normalise both terms to roughly 0..1 so the weight is meaningful,
        # then forbid any pair that fails either gate outright.
        cost = spatial / self.max_distance
        cost += self.color_weight * (color / self.max_color_distance)
        forbidden = (spatial > self.max_distance) | (color > self.max_color_distance)
        cost[forbidden] = 1e6

        rows, cols = linear_sum_assignment(cost)

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e6:      # assignment filled an impossible pair
                continue
            oid = ids[r]
            d = detections[c]
            obj = self.objects[oid]
            obj.update(centroid=d["centroid"], contour=d["contour"],
                       clipped=d["clipped"])
            # Smooth the stored color so a partly-occluded frame doesn't
            # yank a track's appearance away from its true value.
            obj["color"] = 0.8 * obj["color"] + 0.2 * d["color"]
            obj["hits"] += 1
            # Don't let a clipped shape's bogus vertex count pollute the vote.
            if not d["clipped"]:
                obj["history"].append(d["label"])
            self.missing[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r, oid in enumerate(ids):
            if r not in used_rows:
                self._age(oid)

        for c, d in enumerate(detections):
            if c not in used_cols:
                self._register(d["centroid"], d["contour"], d["label"],
                               d["clipped"], d["color"])

        return self.objects

    @staticmethod
    def label_of(obj):
        if not obj["history"]:
            return "?"
        return Counter(obj["history"]).most_common(1)[0][0]


# ------------------------------------------------------------------ drawing


# Palette chosen to stay distinguishable against green and against each other.
PALETTE = [
    (255, 255, 255), (0, 255, 255), (255, 128, 0), (255, 0, 255),
    (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128),
    (200, 200, 0), (0, 165, 255), (255, 0, 128), (128, 255, 255),
]


def color_for(track_id):
    """Stable color per track: the same shape keeps its color across frames."""
    return PALETTE[track_id % len(PALETTE)]


def annotate(frame, objects, fps=None, alpha=0.35):
    out = frame
    # Translucent fills go on first, all in one composite, so that where two
    # shapes overlap you can still see both tints rather than one hiding the
    # other. Outlines are drawn afterwards at full opacity so edges stay crisp.
    if objects:
        overlay = frame.copy()
        for oid, obj in objects.items():
            cv2.drawContours(overlay, [obj["contour"]], -1, color_for(oid), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, out)

    for oid, obj in objects.items():
        color = color_for(oid)
        cx, cy = obj["centroid"]

        cv2.drawContours(out, [obj["contour"]], -1, (0, 0, 0), 6)
        cv2.drawContours(out, [obj["contour"]], -1, color, 3)
        cv2.circle(out, (cx, cy), 7, (0, 0, 0), -1)
        cv2.circle(out, (cx, cy), 5, color, -1)

        label = CentroidTracker.label_of(obj)
        text = f"{label}" + (" (clipped)" if obj["clipped"] else "")
        text += f" ({cx},{cy})"
        for thick, col in ((5, (0, 0, 0)), (2, color)):
            cv2.putText(out, text, (cx - 90, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, thick, cv2.LINE_AA)

    if fps is not None:
        for thick, col in ((5, (0, 0, 0)), (2, (0, 255, 0))):
            cv2.putText(out, f"{fps:.1f} fps | {len(objects)} shapes", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, thick, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------- main


def run(source, output=None, scale=0.5, show=True, limit=None):
    # Ints are camera indices; anything else is a path or a stream URL.
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")

    writer = None
    tracker = CentroidTracker()
    smoothed_fps, frames = None, 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:                     # end of file, or stream dropped
                break

            t0 = time.perf_counter()
            tracker.update(detect(frame, scale), frame)
            objects = tracker.confirmed()
            inst_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            smoothed_fps = inst_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * inst_fps

            vis = annotate(frame, objects, smoothed_fps)

            if output:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(
                        output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                        (vis.shape[1], vis.shape[0]))
                writer.write(vis)

            if show:
                cv2.imshow("shapes", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 60 == 0:
                print(f"frame {frames:5d} | {len(objects)} tracked | {smoothed_fps:.1f} fps")
            if limit and frames >= limit:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"done: {frames} frames processed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="video path, camera index, or stream URL")
    p.add_argument("-o", "--output", help="write annotated video here")
    p.add_argument("-s", "--scale", type=float, default=0.5,
                   help="detection downscale factor (default 0.5)")
    p.add_argument("--no-show", action="store_true", help="headless")
    p.add_argument("--limit", type=int, help="stop after N frames")
    a = p.parse_args()
    run(a.source, a.output, a.scale, not a.no_show, a.limit)