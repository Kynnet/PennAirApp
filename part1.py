
import cv2
import numpy as np


def local_std(gray, k=11):
    #Standard deviation of intensity in a k x k window around each pixel.
    #Var(X) = E[X^2] - E[X]^2
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))


def segment(img, k=11, min_area=500):
    #Return external contours of flat-textured regions.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = local_std(gray, k).astype(np.uint8)

    # Otsu automatically finds optimal threshold
    _, mask = cv2.threshold(std, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use open and close to remove noise in foreground + background
    # high-variance pixels that the window straddles at each shape's edge.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area], mask


def classify(contour):
    """Name a shape from its contour."""
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)

    # Circularity
    if perim > 0 and 4 * np.pi * area / (perim * perim) > 0.85:
        return "circle"

    verts = len(cv2.approxPolyDP(contour, 0.04 * perim, True))
    return {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon"}.get(
        verts, f"{verts}-gon"
    )


def annotate(img, contours):
    out = img.copy()
    for c in contours:
        M = cv2.moments(c)
        #center using moments / area
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        cv2.drawContours(out, [c], -1, (255, 255, 255), 3)
        cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1)

        label = f"{classify(c)} ({cx},{cy})"
        cv2.putText(out, label, (cx - 70, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(out, label, (cx - 70, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "PennAir_2024_App_Static.png"
    img = cv2.imread(path)

    contours, mask = segment(img)
    print(f"found {len(contours)} shapes")
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        M = cv2.moments(c)
        print(f"  {classify(c):15s} area={cv2.contourArea(c):7.0f} "
              f"centroid=({M['m10']/M['m00']:.0f}, {M['m01']/M['m00']:.0f})")

    cv2.imwrite("part1_detected.png", annotate(img, contours))
    cv2.imwrite("part1_mask.png", mask)