import argparse
import time
from collections import Counter, OrderedDict, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([x + w / 2.0, y + h / 2.0])
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def solidity(contour):
    """Area over convex-hull area. Every shape in this clip is convex, so a
    low value means one blob is holding more than one shape."""
    hull = cv2.contourArea(cv2.convexHull(contour))
    return cv2.contourArea(contour) / hull if hull > 0 else 0.0


def hu_moments(contour):
    """Log-scaled Hu moments: a scale- and rotation-invariant signature."""
    m = cv2.HuMoments(cv2.moments(contour)).ravel()
    return np.sign(m) * np.log10(np.abs(m) + 1e-30)


def touches_border(contour, shape, pad=3):
    h, w = shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= pad or y <= pad or x + cw >= w - pad or y + ch >= h - pad


def fill_of(contour, size=None, origin=(0, 0)):
    """Rasterise a contour, either into a given canvas or its own bounding box."""
    if size is None:
        x, y, w, h = cv2.boundingRect(contour)
        canvas = np.zeros((h, w), np.uint8)
        cv2.drawContours(canvas, [contour - [x, y]], -1, 255, -1)
        return canvas, (x, y, w, h)
    canvas = np.zeros(size, np.uint8)
    cv2.drawContours(canvas, [contour - list(origin)], -1, 255, -1)
    return canvas


def overlap(a, b):
    """(intersection, area of a, area of b) in pixels.

    Rasterised into the union of the two bounding boxes rather than a
    full-frame canvas: this runs for every track/blob pair, every frame.
    """
    ax, ay, aw, ah = cv2.boundingRect(a)
    bx, by, bw, bh = cv2.boundingRect(b)
    if ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay:
        return 0.0, cv2.contourArea(a), cv2.contourArea(b)
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    size = (y1 - y0, x1 - x0)
    ma = fill_of(a, size, (x0, y0))
    mb = fill_of(b, size, (x0, y0))
    return (float(cv2.countNonZero(cv2.bitwise_and(ma, mb))),
            float(cv2.countNonZero(ma)), float(cv2.countNonZero(mb)))


def inside_fraction(contour, blob):
    """How much of this outline lies inside that blob."""
    inter, area, _ = overlap(contour, blob)
    return inter / area if area else 0.0


def iou(a, b):
    inter, area_a, area_b = overlap(a, b)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def covered_by(contour, blobs, min_fraction=0.5):
    return any(inside_fraction(contour, b) > min_fraction for b in blobs)


# --------------------------------------------------------------------------
# segmentation: the binary bitmask
# --------------------------------------------------------------------------


def local_std(gray, k):
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))


class Segmenter:

    def __init__(self, history=300, var_threshold=25, learning_rate=0.01,
                 texture_ratio=0.55, texture_reach=21, max_flatness=0.95,
                 protect_ratio=0.90,
                 min_area_frac=0.0015, background_every=10):
        self.model = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.learning_rate = learning_rate
        self.texture_ratio = texture_ratio
        self.protect_ratio = protect_ratio
        self.texture_reach = texture_reach
        self.max_flatness = max_flatness
        self.min_area_frac = min_area_frac
        self.background_every = background_every
        self._background = None
        self.smooth = None
        self.frames = 0

    @property
    def ready(self):
        return self.frames > 20          # let MOG2 see a little history first

    def _motion(self, work, protect):
 
        if not self.ready:
            return self._threshold(self.model.apply(work, learningRate=-1))

        mask = self._threshold(self.model.apply(work, learningRate=0))

        learn = work
        if protect is not None and cv2.countNonZero(protect):
            # Reading the background image back costs 2.2 ms, most of a
            # frame's budget, and it barely moves between frames.
            if self._background is None or self.frames % self.background_every == 0:
                self._background = self.model.getBackgroundImage()
            if self._background is not None:
                learn = work.copy()
                np.copyto(learn, self._background, where=protect[:, :, None] > 0)
        self.model.apply(learn, learningRate=self.learning_rate)
        return mask

    @staticmethod
    def _threshold(mask):
        return (mask > 128).astype(np.uint8) * 255      # drop MOG2's shadows

    def _texture(self, gray):
        window = max(5, int(round(gray.shape[0] / 50)) | 1)   # | 1 forces odd
        std = local_std(gray, window)
        # The median only needs to identify the typical background pixel, and
        # every 4th pixel in each direction says the same thing for a
        # sixteenth of the sort.
        median = float(np.median(std[::4, ::4]))
        mask = (std < self.texture_ratio * median).astype(np.uint8) * 255
        return mask, std, median

    def __call__(self, work, gray, protect=None):
        """Returns (blobs, mask, motion mask, std image, median std)."""
        self.frames += 1
        motion = self._motion(work, protect)
        texture, std, median_std = self._texture(gray)

        # Anchor the texture cue to the motion cue, using it as final adjustment   
        reachable = cv2.dilate(motion, np.ones((self.texture_reach,) * 2, np.uint8))
        mask = cv2.bitwise_or(motion, cv2.bitwise_and(texture, reachable))
        # Remember which pixels look like shape rather than road, so the next
        # frame's conservative update can protect only those (see occupancy).
        self.smooth = (std < self.protect_ratio * median_std).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

        h, w = gray.shape
        min_area = self.min_area_frac * h * w
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [b for b in found
                 if cv2.contourArea(b) > min_area
                 and self._plausible(b, motion, std, median_std)]
        return blobs, mask, motion, std, median_std

    def _plausible(self, blob, motion, std, median_std):
        """Reject what neither cue really supports."""
        fill, (x, y, w, h) = fill_of(blob)
        region = cv2.erode(fill, np.ones((7, 7), np.uint8))
        if cv2.countNonZero(region) < 50:
            region = fill
        area = float(cv2.countNonZero(region))

        # Smoother than its surroundings? A stale patch of background left
        # behind by the motion model scores about 1.0. Real shapes mostly
        # score 0.2-0.8, but the white trapezoid in this clip reaches 0.86 --
        # its gradient is bright enough that compression noise across it
        # rivals the asphalt grain -- so the level has to sit just under 1.0
        # rather than at the bottom of the background's range. Tightening it
        # to 0.85 silently drops that shape: recall 0.91 -> 0.79. Removing
        # the test altogether costs precision instead (0.92 -> 0.77), so it
        # earns its place at this level and not much lower.
        window = std[y:y + h, x:x + w]
        flatness = float(np.mean(window[region > 0])) / max(median_std, 1e-6)
        if flatness > self.max_flatness:
            return False

        # Supported by motion at all? If not, this is the texture cue on its
        # own, which also fires on smooth stretches of road, so demand that it
        # at least look like one of these shapes: convex and whole.
        moved = cv2.countNonZero(cv2.bitwise_and(region, motion[y:y + h, x:x + w]))
        if moved / area < 0.2 and solidity(blob) < 0.9:
            return False
        return True


# --------------------------------------------------------------------------
# optical flow
# --------------------------------------------------------------------------

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def track_features(prev_gray, gray, points, max_error=1.0):
    """Sparse Lucas-Kanade with a forward-backward consistency check.

    Tracking each point forward and then back again, and keeping only the
    points that land where they started, is what makes this usable on shapes
    whose interiors are smooth gradients: the points LK cannot really see
    fail the round trip and are discarded rather than voting.
    """
    if points is None or len(points) < 3:
        return None, None
    forward, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **LK)
    if forward is None:
        return None, None
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, forward, None, **LK)
    if back is None:
        return None, None
    round_trip = np.linalg.norm(points - back, axis=2).ravel()
    good = (status.ravel() == 1) & (status_back.ravel() == 1) & (round_trip < max_error)
    track_features.last_good = good      # which points survived, for callers
                                         # that batched several shapes together
    if good.sum() < 3:
        return None, None
    return forward[good], (forward[good] - points[good]).reshape(-1, 2)


track_features.last_good = None


def seed_features(gray, contour, max_points=60):
    """Corners to follow, taken from inside a shape's own silhouette.

    Eroded, because a feature on the rim sits in a window that is half static
    background, and LK would then report a velocity biased toward zero.
    """
    fill, (x, y, w, h) = fill_of(contour)
    fill = cv2.erode(fill, np.ones((9, 9), np.uint8))
    if cv2.countNonZero(fill) < 100:
        return None
    region = gray[y:y + h, x:x + w]
    points = cv2.goodFeaturesToTrack(region, max_points, 0.01, 5, mask=fill)
    return None if points is None else points + np.float32([x, y])


def split_by_flow(blob, prev_gray, gray, min_area, expected=2, max_k=3,
                  min_separation=1.5, min_part_solidity=0.88, max_samples=1500):
    """Separate overlapping shapes by their velocities.

    Two shapes that have run into each other form one concave blob that no
    amount of morphology cuts apart correctly. But they are only one blob in
    *space*: in the flow field they are two flat regions moving differently,
    so k-means over the flow vectors recovers the partition.

    Dense flow is what makes the per-pixel assignment possible, and it is
    computed on this blob's bounding box alone -- a few percent of the frame,
    a few percent of the 83 ms a whole-frame Farneback costs. Returns None
    when the motions are too alike to trust, which is the caller's cue to
    fall back on the tracks' own memory of where they were.
    """
    x, y, w, h = cv2.boundingRect(blob)
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray[y0:y1, x0:x1], gray[y0:y1, x0:x1],
        None, 0.5, 4, 25, 3, 7, 1.5, 0)

    fill = fill_of(blob, flow.shape[:2], (x0, y0))
    # Sample the interior only: flow on the rim is contaminated by the static
    # background filling the other half of the window.
    inner = cv2.erode(fill, np.ones((7, 7), np.uint8))
    if cv2.countNonZero(inner) < 80:
        return None

    samples = flow[inner > 0].astype(np.float32)
    if len(samples) > max_samples:
        # Fitting the clusters is a question about the distribution, not about
        # every pixel; a fixed-size sample answers it just as well and keeps
        # k-means off the profile's top line.
        samples = samples[np.random.default_rng(0).choice(
            len(samples), max_samples, replace=False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    ys, xs = np.nonzero(fill)
    vectors = flow[ys, xs]

    # Try the number of shapes the tracker believes are in here first.
    order = sorted(range(2, max_k + 1), key=lambda k: (k != expected, -k))
    for k in order:
        if len(samples) < 20 * k:
            continue
        _, _, centers = cv2.kmeans(samples, k, None, criteria, 5,
                                   cv2.KMEANS_PP_CENTERS)
        gaps = [np.linalg.norm(centers[i] - centers[j])
                for i in range(k) for j in range(i + 1, k)]
        if min(gaps) < min_separation:
            continue

        assign = np.argmin(
            np.linalg.norm(vectors[:, None, :] - centers[None], axis=2), axis=1)
        parts = []
        for cluster in range(k):
            piece = np.zeros(fill.shape, np.uint8)
            piece[ys[assign == cluster], xs[assign == cluster]] = 255
            piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            piece = cv2.morphologyEx(piece, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            found, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            parts += [f + [x0, y0] for f in found if cv2.contourArea(f) > min_area]

        # Only believe a split that produced convex pieces of a sane size.
        if len(parts) >= 2 and all(solidity(p) >= min_part_solidity for p in parts):
            return parts
    return None


CARVE_FLOOR_MULTIPLE = 3.0
CARVE_FLOOR_FRACTION = 0.6


def carve(blob, known, min_area, min_solidity=0.92, margin=9):
    """Whatever a merged blob holds that the known shapes do not explain.

    When two shapes meet and the flow split fails, the tracks standing in the
    blob draw themselves from memory -- but a shape that arrives already
    touching another has no memory to draw from, and stays invisible for as
    long as the pair stay together. Subtracting the outlines that *are*
    accounted for leaves the newcomer behind.

    Two details keep this honest. A margin is erased around each known
    outline, because those outlines are predictions rather than measurements
    and a shape moving 20 px per frame otherwise leaves a crescent of
    "unexplained" pixels along its leading edge -- convex, shape-sized, and
    duly promoted into a phantom drawn inside a real shape. And the leftover
    must be solid: the residue of a genuinely occluded shape has a bite out
    of it and is rejected, so only a whole shape that happens to be touching
    another is promoted.
    """
    x, y, w, h = cv2.boundingRect(blob)
    mask = fill_of(blob, (h, w), (x, y))
    for contour in known:
        shifted = contour - [x, y]
        cv2.drawContours(mask, [shifted], -1, 0, -1)
        cv2.drawContours(mask, [shifted], -1, 0, margin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [f + [x, y] for f in found
            if cv2.contourArea(f) > min_area and solidity(f) >= min_solidity]


def carve_floor(min_area, known):
    """How big a leftover must be before it counts as a shape.

    The plain detection floor is far too generous: the sliver of a trapezoid
    poking out from behind a pentagon clears it easily and becomes a phantom
    drawn inside a real shape. A newcomer has to be in the same size class as
    the shapes already being tracked.
    """
    if known:
        smallest = min(cv2.contourArea(c) for c in known)
        return max(CARVE_FLOOR_MULTIPLE * min_area, CARVE_FLOOR_FRACTION * smallest)
    return CARVE_FLOOR_MULTIPLE * min_area


def relocate(contour, mask, edges, guess, radius, edge_weight=0.7, prior_weight=0.5):
    """Re-find a known outline in the current bitmask.

    This is what survives overlap. Correlating a shape's own silhouette
    against the bitmask asks "where could this outline sit and be entirely
    foreground?", and an occluded shape's pixels are still foreground -- the
    occluder is standing on them. Containment alone is ambiguous, so two more
    terms pin it down: the template's edge map against the mask's edge map,
    which locks whatever part of the rim is still exposed, and a prior
    pulling the answer toward where optical flow says the shape went.

    Returns (offset, containment) or None if it cannot be matched.
    """
    template, (x, y, w, h) = fill_of(contour)
    area = float(cv2.countNonZero(template))
    if w < 6 or h < 6 or area < 25:
        return None

    gx, gy = int(round(guess[0])), int(round(guess[1]))
    height, width = mask.shape
    x0, y0 = max(0, x + gx - radius), max(0, y + gy - radius)
    x1 = min(width, x + gx + w + radius)
    y1 = min(height, y + gy + h + radius)
    if x1 - x0 < w + 1 or y1 - y0 < h + 1:
        return None

    tmpl = (template > 0).astype(np.float32)
    edge = cv2.morphologyEx(template, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
    fill_score = cv2.matchTemplate(mask[y0:y1, x0:x1], tmpl, cv2.TM_CCORR) / area
    edge_score = cv2.matchTemplate(edges[y0:y1, x0:x1], edge,
                                   cv2.TM_CCORR) / max(float(edge.sum()), 1.0)

    rows, cols = np.indices(fill_score.shape)
    stray = np.hypot((x0 + cols - x) - gx, (y0 + rows - y) - gy) / max(radius, 1)
    total = fill_score + edge_weight * edge_score - prior_weight * stray

    i, j = np.unravel_index(np.argmax(total), total.shape)
    return (np.array([x0 + j - x, y0 + i - y], np.float32),
            float(fill_score[i, j]))


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

EDGES_OF = {"triangle": 3, "square": 4, "rectangle": 4, "trapezoid": 4,
            "parallelogram": 4, "quadrilateral": 4, "pentagon": 5,
            "hexagon": 6, "heptagon": 7, "octagon": 8}


def classify(contour):
    """Name a shape from its outline alone."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0 or area <= 0:
        return "?"

    # Circularity: 4*pi*A/P^2 is 1 for a circle and falls away with corners.
    if 4 * np.pi * area / (perimeter * perimeter) > 0.85:
        return "circle"

    poly = cv2.approxPolyDP(contour, 0.04 * perimeter, True).reshape(-1, 2)
    if len(poly) < 3:
        # approxPolyDP can collapse a sliver to two points, and "2-gon" is not
        # a shape. Say nothing rather than something impossible.
        return "?"
    if len(poly) == 4:
        return name_quadrilateral(poly)
    return {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon",
            8: "octagon"}.get(len(poly), f"{len(poly)}-gon")


def name_quadrilateral(poly, angle_tol=8.0, side_tol=0.15):
    """Tell a rectangle from a trapezoid -- this clip contains both.

    The angle tolerance is deliberately tight: this clip's trapezoid has
    interior angles near 78 degrees, only 12 off square, so a loose tolerance
    calls it a rectangle. Being strict costs nothing, because the parallel
    side tests below name it correctly anyway.
    """
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() <= 0:
        return "quadrilateral"
    units = edges / lengths[:, None]

    interior = [np.degrees(np.arccos(np.clip(-np.dot(units[i - 1], units[i]), -1, 1)))
                for i in range(4)]
    if all(abs(angle - 90) < angle_tol for angle in interior):
        ratio = lengths[0] / lengths[1]
        return "square" if abs(ratio - 1) < side_tol else "rectangle"

    cos_tol = np.cos(np.radians(angle_tol))
    pairs = sum(1 for i in (0, 1) if abs(np.dot(units[i], units[i + 2])) > cos_tol)
    return {2: "parallelogram", 1: "trapezoid"}.get(pairs, "quadrilateral")


def outline_of(contour, label):
    """The polygon to draw: straight edges for a polygon, the contour itself
    for a circle.

    When the track has settled on a name, the polygon is fitted to the number
    of edges that name implies, by searching for the approxPolyDP tolerance
    that yields exactly that many vertices. Otherwise a shape can end up
    labelled "trapezoid" with six vertices drawn around it, which reads as a
    bug even when the name is right.
    """
    if label == "circle":
        return contour
    perimeter = cv2.arcLength(contour, True)
    wanted = EDGES_OF.get(label)
    if wanted:
        for eps in np.linspace(0.01, 0.10, 19):
            poly = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(poly) == wanted:
                return poly
    poly = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    return poly if 3 <= len(poly) <= 10 else contour


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------


MERGED_SOLIDITY = 0.85


SMOOTHABLE_SOLIDITY = 0.85
SMOOTH_EPS = 0.01


def smooth(contour):
    """Take the pixel staircase off a contour, for drawing only.

    Every shape in this clip is convex, so a nearly convex contour is one
    shape with mask noise along its edge, and the staircase is worth removing:
    it is what makes an outline look hand-drawn, and a deep enough notch drops
    solidity below the merge threshold so a lone triangle gets labelled
    "merged shapes" and drawn as a raw jagged outline.

    Applied when the outline is drawn, never to the contour a track stores.
    Two things were tried on the stored contour and both were worse. The
    convex hull can only add area, so it pushed outlines past the shapes they
    described (precision 0.968 -> 0.946) and, by making everything convex,
    destroyed the solidity signal the merge tests read. approxPolyDP does not
    invent area but it shaves corners inward, costing recall (0.953 -> 0.941)
    for no gain anywhere else. Measurement wants the contour exactly as
    found; only the picture wants it tidy.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0 or solidity(contour) < SMOOTHABLE_SOLIDITY:
        return contour
    return cv2.approxPolyDP(contour, SMOOTH_EPS * perimeter, True)


def describe(contour, frame_shape, whole):
    """A detection: everything the tracker may know about a blob.

    `whole` says this contour is one entire isolated shape, as opposed to a
    piece carved out of a merged blob. Only a whole, unclipped silhouette is
    allowed to define what a track looks like or to vote on its name.
    """
    clipped = touches_border(contour, frame_shape)
    return {
        "contour": contour,
        "centroid": centroid(contour),
        "area": cv2.contourArea(contour),
        "hu": hu_moments(contour),
        "label": classify(contour),
        "clipped": clipped,
        "partial": not whole,
        "votable": whole and not clipped,
    }


class ShapeTracker:
    """Multi-object tracker on geometry and motion alone.

    Detections are assigned to tracks by a global (Hungarian) solve over a
    cost built from optical-flow-predicted distance, area ratio and
    Hu-moment distance -- no appearance term anywhere.

    A track that wins no detection is not presumed lost. If it is sitting on
    foreground that another shape has already claimed it is *overlapped*:
    drawn from memory, flagged, and carried forward on its own optical flow.
    Only a track with nowhere to be starts ageing out.
    """

    def __init__(self, max_distance=60, max_missing=15, max_coasting=45,
                 vote_window=21, min_hits=1, area_weight=1.0, shape_weight=0.6,
                 keep_containment=0.80, duplicate_iou=0.55,
                 merged_area_ratio=1.5):
        self.objects = OrderedDict()
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.max_coasting = max_coasting
        self.vote_window = vote_window
        self.min_hits = min_hits
        self.area_weight = area_weight
        self.shape_weight = shape_weight
        self.keep_containment = keep_containment
        self.duplicate_iou = duplicate_iou
        self.merged_area_ratio = merged_area_ratio
        # Areas of whole, isolated, unclipped shapes, so that "this outline is
        # too big to be one shape" has something to mean.
        self.shape_areas = deque(maxlen=100)

    # -- inspection --------------------------------------------------------

    def visible(self):
        """Tracks solid enough to draw, and on screen right now.

        Each is told whether its own outline can still be believed to be a
        single shape. Two tests, because either alone misses cases: a
        non-convex outline is obviously more than one shape, and an outline
        half again bigger than the shapes this clip actually contains is too,
        however convex it looks -- a circle resting on a triangle is quite
        convex, and would otherwise be confidently named a trapezoid.
        """
        typical = self.typical_area
        live = OrderedDict((i, o) for i, o in self.objects.items()
                           if o["hits"] >= self.min_hits and o["missing"] == 0)
        for obj in live.values():
            area = cv2.contourArea(obj["contour"])
            obj["merged"] = (solidity(obj["contour"]) < MERGED_SOLIDITY
                             or (typical is not None
                                 and area > self.merged_area_ratio * typical))

        # An unresolved merge is a fallback for when nothing better is known.
        # Once the shapes inside it are individually outlined, the region
        # drawn around the pair is worse than redundant: it is a second
        # outline sprawling across both of them, which is most of what
        # "detection extends past the shape" looks like on screen.
        singles = [o["contour"] for o in live.values() if not o["merged"]]
        for i in [i for i, o in live.items()
                  if o["merged"] and singles and self._explained(o["contour"], singles)]:
            del live[i]
        return live

    @staticmethod
    def _explained(contour, singles, min_fraction=0.55):
        """Is this region already accounted for by individual outlines?"""
        area = cv2.contourArea(contour)
        return area > 0 and sum(overlap(contour, s)[0] for s in singles) / area > min_fraction

    @property
    def typical_area(self):
        return float(np.median(self.shape_areas)) if len(self.shape_areas) >= 5 else None

    def _single_shape(self, area):
        """Is this one shape's worth of pixels, by this clip's standards?"""
        typical = self.typical_area
        return typical is None or area < self.merged_area_ratio * typical

    def outlines(self):
        """Where each live track expects to be in the frame about to arrive.

        Dead-reckoned one frame forward, because the next frame's
        segmentation consumes this: comparing new blobs against outlines that
        are one frame stale misjudges which blobs are merges.
        """
        return [o["contour"] + np.round(o["velocity"]).astype(np.int32)
                for o in self.objects.values() if o["missing"] == 0]

    def occupancy(self, shape, smooth=None):
        """Where the tracks are, so the background model can leave them be.

        Restricted to pixels that actually look like shape. A track's outline
        can enclose more than the shape -- most often the ghost the model
        leaves behind at the position a shape has just left, which reads as
        foreground over bare road and gets swallowed into the same blob. If
        the whole outline is protected, that ghost is never relearned and the
        track keeps drawing a staircase of shape-plus-ghost for as long as it
        lives.

        Protecting only the smooth pixels breaks the loop precisely: the real
        shape keeps its protection and is never absorbed, while the ghost is
        left exposed and washes out on its own. Withholding protection from
        whole tracks instead was tried, and it does clear the ghosts, but it
        also feeds genuinely overlapped shapes to the model (recall 0.952 ->
        0.913).
        """
        mask = np.zeros(shape[:2], np.uint8)
        for obj in self.objects.values():
            if obj["missing"] == 0:
                cv2.drawContours(mask, [obj["contour"]], -1, 255, -1)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        return mask if smooth is None else cv2.bitwise_and(mask, smooth)

    @staticmethod
    def label_of(obj):
        if not obj["history"]:
            # Nothing has voted yet -- a track born from a shape halfway off
            # the edge of frame, say. Name what is visible; the vote takes
            # over once the whole shape has been seen.
            return classify(obj["contour"])
        return Counter(obj["history"]).most_common(1)[0][0]

    # -- the frame update --------------------------------------------------

    def update(self, detections, ambiguous, mask, prev_gray, gray):
        self._edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                       np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
        self._bitmask = (mask > 0).astype(np.float32)
        self._gray = gray

        self._flow(prev_gray, gray)

        if not self.objects:
            for det in detections:
                self._register(det, gray)
            return

        ids = list(self.objects)
        matched_tracks, matched_dets = set(), set()
        if detections:
            cost = np.array([[self._cost(self.objects[oid], det) for det in detections]
                             for oid in ids])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e6:
                    self._absorb(ids[r], detections[c], gray)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # Somewhere a shape could be hiding: a blob nothing could split, or
        # one another track won that is too big to be a single shape. The size
        # test is the important half -- without it any blob another track just
        # matched counted as cover, so a track that had lost its shape could
        # drift onto an ordinary, fully explained shape and sit there
        # redrawing itself.
        occupied = list(ambiguous) + [
            detections[c]["contour"] for c in matched_dets
            if not self._single_shape(detections[c]["area"])]

        for r, oid in enumerate(ids):
            if r in matched_tracks:
                continue
            obj = self.objects[oid]
            self._refine(obj)
            moved = obj["contour"] + np.round(obj["shift"]).astype(np.int32)
            if obj["containment"] >= self.keep_containment and covered_by(moved, occupied):
                self._coast(oid, moved)
            else:
                self._miss(oid)

        placed = [o["contour"] for o in self.objects.values() if o["missing"] == 0]
        for c, det in enumerate(detections):
            if c in matched_dets or det["partial"]:
                # A piece carved out of a merged blob is not evidence of a new
                # shape, only of one already tracked. A *clipped* blob is a
                # new shape halfway into frame, and does get to start a track.
                continue
            # Nor is a blob that an existing shape is sitting inside: two
            # shapes that have just met form a region convex enough to pass as
            # one clean detection, whose IoU with either half is only ~0.5 --
            # under any sane duplicate threshold. Containment states the
            # objection that IoU cannot.
            if any(iou(det["contour"], p) > self.duplicate_iou
                   or inside_fraction(p, det["contour"]) > 0.7 for p in placed):
                continue
            self._register(det, gray)

        self._dedupe({ids[r] for r in matched_tracks})

    # -- the pieces --------------------------------------------------------

    def _flow(self, prev_gray, gray):
        """Where has each shape gone? One batched Lucas-Kanade call.

        Every track's feature points go into a single calcOpticalFlowPyrLK
        call, so the image pyramids are built once per frame instead of once
        per track -- with a handful of shapes on screen that is most of the
        cost of this step.
        """
        for obj in self.objects.values():
            obj["shift"] = obj["velocity"]
            obj["predicted"] = obj["centroid"] + obj["shift"]
            obj["containment"] = None            # computed lazily, see _refine

        if prev_gray is None:
            return
        owners, batch = [], []
        for oid, obj in self.objects.items():
            if obj["features"] is not None and len(obj["features"]) >= 3:
                owners += [oid] * len(obj["features"])
                batch.append(obj["features"])
        if not batch:
            return

        points = np.concatenate(batch).astype(np.float32)
        moved, deltas = track_features(prev_gray, gray, points)
        if deltas is None:
            return
        # track_features drops the points that failed, so recover which track
        # each survivor came from before taking per-shape medians.
        kept = np.array(owners)[track_features.last_good]
        for oid, obj in self.objects.items():
            mine = kept == oid
            if mine.sum() < 3:
                obj["features"] = None
                continue
            velocity = np.median(deltas[mine], axis=0)
            # Drop points that disagree with the consensus: they are on the
            # occluder, not on this shape.
            agree = np.linalg.norm(deltas[mine] - velocity, axis=1) < 2.0
            survivors = moved[mine][agree] if agree.sum() >= 3 else moved[mine]
            obj["features"] = survivors
            obj["shift"] = velocity
            obj["predicted"] = obj["centroid"] + velocity

    def _refine(self, obj):
        """Pin the flow estimate down against the bitmask.

        Only worth doing for a track the assignment could not satisfy: LK
        points on a hidden part of a shape are gone, and the silhouette still
        knows where the rim is. Doing it for every track every frame was a
        third of the tracker's time for no benefit to the tracks that a
        detection was going to match anyway.
        """
        if obj["containment"] is not None:
            return
        radius = int(np.clip(12 + 0.5 * np.linalg.norm(obj["shift"])
                             + 6 * obj["missing"], 12, 60))
        found = relocate(obj["contour"], self._bitmask, self._edges,
                         obj["shift"], radius)
        if found is None:
            obj["containment"] = 0.0
        else:
            obj["shift"], obj["containment"] = found
            obj["predicted"] = obj["centroid"] + obj["shift"]

    def _cost(self, obj, det):
        # All distances are in working-scale pixels. The cap is only a few
        # frames of travel on purpose: a generous one lets a track hop onto a
        # different shape, and a track wearing another shape's name is worse
        # than a track that has to start over.
        distance = np.linalg.norm(obj["predicted"] - det["centroid"])
        ratio = np.log(max(det["area"], 1.0) / max(obj["area"], 1.0))
        # A shape cannot teleport, and cannot grow half again its own size --
        # that would be two shapes. It can *shrink* without limit, by sliding
        # off the frame edge or disappearing behind something, so that side of
        # the gate stays soft.
        typical = self.typical_area
        undersized = typical is not None and obj["area"] < 0.8 * typical
        if distance > self.max_distance or ratio > (1.6 if (obj["clipped"] or undersized) else 0.4):
            return 1e6
        # Hu distance is priced into the cost rather than used as a veto:
        # vetoing on it (or on the classified name) makes a settled track
        # refuse its own shape on frames where the silhouette is briefly
        # imperfect, which measured out as triple the identity churn.
        shape = np.linalg.norm(obj["hu"][:3] - det["hu"][:3])
        return (distance / self.max_distance
                + self.area_weight * abs(ratio)
                + self.shape_weight * min(shape, 3.0) / 3.0)

    def _register(self, det, gray):
        self.objects[self.next_id] = {
            "contour": det["contour"], "centroid": det["centroid"],
            "area": det["area"], "hu": det["hu"], "velocity": np.zeros(2),
            "features": seed_features(gray, det["contour"]),
            "history": deque([det["label"]] if det["votable"] else [],
                             maxlen=self.vote_window),
            "hits": 1, "missing": 0, "coasted": 0,
            "overlapped": False, "clipped": det["clipped"],
            "shift": np.zeros(2), "containment": 1.0,
            "predicted": det["centroid"], "merged": False,
        }
        if det["votable"] and self._single_shape(det["area"]):
            self.shape_areas.append(det["area"])
        self.next_id += 1

    def _absorb(self, oid, det, gray):
        """Fold a detection into a track.

        A detection may always tell a track *where* it is; only a whole,
        isolated, single-shape silhouette may tell it what it *looks like*.

        That second test measures the detection against `typical_area` -- how
        big the shapes in this clip actually are -- and deliberately not
        against what this track remembers its own size to be. Comparing
        against the track's own memory looks equivalent and is not: it has no
        way back. One bad silhouette (a merged pair, or a shape caught
        halfway off the frame edge) writes a wrong remembered area, and from
        then on the correct detection disagrees with that wrong memory and is
        refused for disagreeing -- so the track keeps redrawing a stale
        outline under the wrong name for as long as it lives. Measured, that
        locked one track out for 129 consecutive frames while a perfectly
        good detection sat underneath it every one of them.
        """
        obj = self.objects[oid]
        trustworthy = det["votable"] and self._single_shape(det["area"])

        if trustworthy:
            obj["velocity"] = 0.5 * obj["velocity"] + 0.5 * (det["centroid"] - obj["centroid"])
            obj["centroid"] = det["centroid"]
            obj["contour"] = det["contour"]
            obj["area"] = det["area"]
            obj["hu"] = det["hu"]
            obj["clipped"] = det["clipped"]
            obj["overlapped"] = False
            obj["history"].append(det["label"])
            obj["features"] = seed_features(gray, det["contour"])
            self.shape_areas.append(det["area"])
            obj["coasted"] = 0

        elif det["clipped"] and not det["partial"]:
            # A shape halfway out of frame is never "votable", so it used to
            # fall through to the memory branch and be drawn at full size,
            # lagging behind, with half the outline lying on bare road.
            # Relocation cannot rescue that -- matching a whole template
            # against a half-visible shape pulls the answer inward. The
            # measured silhouette is the honest visible extent, so it draws
            # the outline while the remembered size and name keep identifying
            # the track.
            obj["velocity"] = 0.5 * obj["velocity"] + 0.5 * (det["centroid"] - obj["centroid"])
            obj["centroid"] = det["centroid"]
            obj["contour"] = det["contour"]
            obj["clipped"] = True
            obj["overlapped"] = False
            obj["features"] = seed_features(gray, det["contour"])
            obj["coasted"] += 1

        else:
            self._refine(obj)
            # Keeping a remembered outline needs two things to be true: it
            # still fits the image, and there is something here with room to
            # hide a shape inside. Containment alone cannot tell "I am behind
            # that" from "I have drifted on top of that".
            cover = det["partial"] or not self._single_shape(det["area"])
            if obj["containment"] >= self.keep_containment and cover:
                obj["contour"] = obj["contour"] + np.round(obj["shift"]).astype(np.int32)
                obj["centroid"] = obj["centroid"] + obj["shift"]
                obj["velocity"] = 0.6 * obj["velocity"] + 0.4 * obj["shift"]
            else:
                # The remembered outline no longer sits on anything. Keeping
                # it draws a shape where there is none; this path had no
                # quality gate at all and was the largest single source of
                # stray outlines. The measurement may be partial, so it still
                # does not vote or redefine the track's size, but it is at
                # least really there.
                obj["contour"] = det["contour"]
                obj["centroid"] = det["centroid"]
                obj["velocity"] = 0.5 * obj["velocity"] + 0.5 * obj["shift"]
                obj["features"] = seed_features(gray, det["contour"])
            obj["overlapped"] = True
            # The coasting budget counts frames since this track was last
            # properly measured, not frames since it last matched anything.
            obj["coasted"] += 1

        obj["missing"] = 0
        obj["hits"] += 1
        if obj["coasted"] > self.max_coasting:
            del self.objects[oid]

    def _coast(self, oid, moved):
        """Carry an overlapped track forward on its own flow."""
        obj = self.objects[oid]
        obj["contour"] = moved
        obj["centroid"] = obj["centroid"] + obj["shift"]
        obj["velocity"] = 0.6 * obj["velocity"] + 0.4 * obj["shift"]
        obj["overlapped"] = True
        obj["missing"] = 0
        # Deliberately not a hit. Hits count frames backed by a real
        # detection: a stranded outline lying across a genuine shape can coast
        # indefinitely, and if that inflated its hit count it would outrank
        # the real detection's track in _dedupe and evict it, frame on frame.
        obj["coasted"] += 1
        if obj["coasted"] > self.max_coasting:
            del self.objects[oid]

    def _miss(self, oid):
        obj = self.objects[oid]
        # Tell "hidden" from "not there". A track whose outline still lands on
        # foreground is plausibly behind something and worth waiting for; one
        # landing on bare road has lost its shape, and keeping it alive only
        # strands it further away.
        obj["missing"] += 1 if (obj["containment"] or 0.0) >= 0.5 else 4
        obj["overlapped"] = False
        # Keep dead-reckoning so an identity survives a shape passing behind
        # another, but damp it: at 20 px per frame, coasting at full speed for
        # a second puts the outline off the far side of the frame, from where
        # it can never find its way home.
        obj["velocity"] = 0.75 * obj["velocity"]
        obj["centroid"] = obj["centroid"] + obj["velocity"]
        obj["contour"] = obj["contour"] + np.round(obj["velocity"]).astype(np.int32)
        obj["features"] = None
        if obj["missing"] > self.max_missing:
            del self.objects[oid]

    def _dedupe(self, matched):
        """Two tracks on one shape: keep the better-established one.

        The test is IoU and only IoU. Containment looks like the sharper
        instrument -- a duplicate does sit almost entirely inside its twin --
        but it is wrong here: a small shape passing behind a large one is also
        almost entirely contained, and dropping it costs far more than the
        duplicates are worth (measured: recall 0.85 -> 0.78, and triple the
        churn). A track a detection actually backed this frame outranks one
        that is merely coasting.
        """
        live = [i for i, o in self.objects.items() if o["missing"] == 0]
        rank = lambda i: (i in matched, self.objects[i]["hits"], -i)
        drop = set()
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                ia, ib = live[a], live[b]
                if ia in drop or ib in drop:
                    continue
                if iou(self.objects[ia]["contour"], self.objects[ib]["contour"]) \
                        > self.duplicate_iou:
                    drop.add(ia if rank(ia) < rank(ib) else ib)
        for oid in drop:
            del self.objects[oid]


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

PALETTE = [(255, 255, 255), (0, 255, 255), (255, 128, 0), (255, 0, 255),
           (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128),
           (200, 200, 0), (0, 165, 255), (255, 0, 128), (128, 255, 255)]


def color_for(track_id):
    """Stable per track: a shape keeps its color for its whole life."""
    return PALETTE[track_id % len(PALETTE)]


def draw_text(image, text, origin, color, scale=0.6):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def annotate(frame, objects, fps=None):
    for oid, obj in objects.items():
        color = color_for(oid)
        # Set by the tracker: this outline holds more than one shape. It is
        # still drawn -- an outline in the right place is far more useful
        # than nothing -- but naming it would be inventing an answer, so it
        # says so instead.
        merged = obj.get("merged", False)
        label = ShapeTracker.label_of(obj)
        poly = smooth(obj["contour"]) if merged else outline_of(obj["contour"], label)
        points = poly.reshape(-1, 2)

        # If the outline could not be fitted to the number of edges the voted
        # name implies, the vote does not describe the silhouette on screen --
        # so believe the silhouette. Otherwise the label and the shape drawn
        # around it contradict each other, e.g. "trapezoid - 5 edges".
        if not merged and EDGES_OF.get(label) not in (None, len(points)):
            label = classify(obj["contour"])
            poly = outline_of(obj["contour"], label)
            points = poly.reshape(-1, 2)
        cx, cy = np.round(obj["centroid"]).astype(int)

        # Edges: the polygon the classifier actually measured.
        cv2.polylines(frame, [poly], True, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, color, 3, cv2.LINE_AA)

        # Vertices, so the edge count is shown rather than just asserted.
        if label != "circle" and not merged:
            for px, py in points:
                cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)

        cv2.drawMarker(frame, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 24, 5, cv2.LINE_AA)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        if merged:
            label, edges = "merged shapes", "edges unresolved"
        elif label == "circle":
            edges = "1 curved edge"
        else:
            edges = f"{len(points)} edges"
        flags = ("" if not obj["overlapped"] else " [overlap]") + \
                ("" if not obj["clipped"] else " [clipped]")
        draw_text(frame, f"#{oid} {label} - {edges}{flags}",
                  (int(points[:, 0].min()) - 5, int(points[:, 1].min()) - 14),
                  color, 0.62)
        draw_text(frame, f"({cx}, {cy})", (cx + 14, cy + 6), color, 0.55)

    if fps is not None:
        draw_text(frame, f"{fps:4.1f} fps | {len(objects)} shapes", (20, 45),
                  (0, 255, 0), 0.9)
    return frame


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


class Pipeline:
    """Segment, split, track -- one frame at a time."""

    def __init__(self, scale=0.5, clean_solidity=0.90, min_area_frac=0.0015,
                 claim_unowned=True, max_flatness=0.95):
        self.scale = scale
        self.clean_solidity = clean_solidity
        self.claim_unowned = claim_unowned
        self.min_area_frac = min_area_frac
        self.segmenter = Segmenter(min_area_frac=min_area_frac,
                                   max_flatness=max_flatness)
        self.tracker = ShapeTracker()
        self.prev_gray = None
        self.counts = (0, 0, 0)

    def __call__(self, frame):
        work = frame if self.scale == 1.0 else cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        height, width = work.shape[:2]
        min_area = self.min_area_frac * height * width

        blobs, mask, _, _, _ = self.segmenter(
            work, gray,
            protect=self.tracker.occupancy(work.shape, self.segmenter.smooth))
        if not self.segmenter.ready:
            self.prev_gray = gray
            return OrderedDict(), mask

        detections, ambiguous = [], []
        live = self.tracker.outlines()
        for blob in blobs:
            # Concavity is the obvious sign that two shapes have run into each
            # other, but not a reliable one -- a circle on a triangle makes a
            # convex blob. So ask the tracks too: if two are standing in this
            # blob and it is bigger than either, it is a merge however convex
            # it looks. Without this test a merged blob gets absorbed as one
            # shape, doubling that track's remembered area and killing it the
            # moment the two separate again.
            owners = [c for c in live if inside_fraction(c, blob) > 0.6]
            crowded = (len(owners) >= 2 and cv2.contourArea(blob)
                       > 1.2 * max(cv2.contourArea(c) for c in owners))
            if not crowded and solidity(blob) >= self.clean_solidity:
                detections.append(describe(blob, work.shape, whole=True))
                continue

            parts = (split_by_flow(blob, self.prev_gray, gray, min_area * 0.5,
                                   expected=max(2, len(owners)))
                     if self.prev_gray is not None else None)
            if parts:
                detections += [describe(p, work.shape, whole=False) for p in parts]
            elif owners:
                # Not offered as a detection: the tracks inside it will
                # relocate their own outlines against it, which is far more
                # accurate than one contour drawn around the pair. Anything
                # in there they cannot account for is a shape not yet met.
                ambiguous.append(blob)
                detections += [describe(r, work.shape, whole=True)
                               for r in carve(blob, live,
                                              carve_floor(min_area, live))]
            else:
                # Nothing claims this blob and flow could not divide it, so
                # there is no better hypothesis available than "one shape".
                # Staying silent here is what left concave blobs undrawn:
                # measured, it is the difference between covering 72% and 87%
                # of true shape pixels. If it really is two shapes, the split
                # will separate them as soon as they move differently.
                ambiguous.append(blob)
                if self.claim_unowned:
                    # whole=True so it can start a track and be drawn, but it
                    # must not vote on a name or count towards what a shape's
                    # size is: it may well be two shapes.
                    claimed = describe(blob, work.shape, whole=True)
                    claimed["votable"] = False
                    detections.append(claimed)

        self.tracker.update(detections, ambiguous, mask, self.prev_gray, gray)
        self.prev_gray = gray
        self.counts = (len(blobs), len(detections), len(ambiguous))
        return self.tracker.visible(), mask

    def scaled_to_frame(self, objects):
        """Move outlines from working scale back to full resolution."""
        return OrderedDict(
            (oid, dict(obj,
                       contour=np.round(obj["contour"] / self.scale).astype(np.int32),
                       centroid=obj["centroid"] / self.scale))
            for oid, obj in objects.items())


def run(source, output=None, scale=0.5, show=True, limit=None, debug=False):
    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"could not open {source}")

    pipeline = Pipeline(scale=scale)
    writer, smoothed_fps, frames = None, None, 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()
            objects, mask = pipeline(frame)
            elapsed = time.perf_counter() - started
            instant = 1.0 / max(elapsed, 1e-6)
            smoothed_fps = instant if smoothed_fps is None \
                else 0.9 * smoothed_fps + 0.1 * instant

            view = annotate(frame, pipeline.scaled_to_frame(objects), smoothed_fps)
            if debug and mask is not None:
                inset = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), None,
                                   fx=0.25 / scale, fy=0.25 / scale)
                view[-inset.shape[0]:, -inset.shape[1]:] = inset

            if output:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (view.shape[1], view.shape[0]))
                writer.write(view)
            if show:
                cv2.imshow("part3", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 60 == 0:
                blobs, dets, merged = pipeline.counts
                print(f"frame {frames:5d} | {blobs} blobs {dets} detections "
                      f"{merged} unsplit | {len(objects)} tracked "
                      f"| {smoothed_fps:.1f} fps")
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
    parser = argparse.ArgumentParser(description="motion-based shape detection")
    parser.add_argument("source", help="video path, camera index, or stream URL")
    parser.add_argument("-o", "--output", help="write the annotated video here")
    parser.add_argument("-s", "--scale", type=float, default=0.5,
                        help="detection downscale factor (default 0.5)")
    parser.add_argument("--no-show", action="store_true", help="headless")
    parser.add_argument("--limit", type=int, help="stop after N frames")
    parser.add_argument("--debug", action="store_true", help="inset the bitmask")
    args = parser.parse_args()
    run(args.source, args.output, args.scale, not args.no_show, args.limit, args.debug)
