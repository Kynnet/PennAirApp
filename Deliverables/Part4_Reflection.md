# Approach

Part 4 imports part 3 and then adds the math on top for the annotation.

The pinhole model says a point at $(X, Y, Z)$ in front of the camera lands at

$$u = f_x \frac{X}{Z} + c_x \qquad\qquad v = f_y \frac{Y}{Z} + c_y$$

Running that backwards, a circle of radius $R$ sitting at depth $Z$ images as an ellipse whose semi-axes are

$$a = f_x \frac{R}{Z} \qquad\qquad b = f_y \frac{R}{Z}$$

The area of an ellipse is $\pi a b$, so the circle covers

$$A = \pi \, \frac{f_x f_y R^2}{Z^2}$$

pixels. Every term in that is known except the depth, which rearranges to

$$Z = R \sqrt{\frac{\pi f_x f_y}{A}}$$

Area should be more accurate since it is more of an average than diameter. It also avoids having to decide which axis to measure along, since the two focal lengths are not quite equal.

Inverting the projection then gives each center in inches:

$$X = (u - c_x)\frac{Z}{f_x} \qquad\qquad Y = (v - c_y)\frac{Z}{f_y}$$

# Challenges

The easiest mistake to make silently was mixing up resolutions. Detection runs at half scale for speed, but the intrinsics describe the full resolution image. The contours had to be scaled back to full resolution before any of the geometry happens.

When the circle slides off the edge of the frame or another shape covers part of it, its visible area shrinks, and since $Z \propto 1/\sqrt{A}$ a smaller circle reads as a more distant one. I solved this byall, and depth is the median of recent estimates.
When the circle slides off the edge of the frame or another shape covers part of it, its visible area shrinks, and since $Z \propto 1/\sqrt{A}$ a smaller circle reads as a more distant one. I solved this by not using clipped circles for measurement, and to increase accuracy, I used the median of the measurements.