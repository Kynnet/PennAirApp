# Approach

Use the standard deviation colors of a kernel to estimate the texture around pixels. Shapes have smooth texture and background is more noisy, thus running a binary bit mask on this standard deviation produces the shapes.

Using MorphEx open and close removes noise from these shapes, then countouring on the bitmask should produce the shapes

Using OpenCV to estimate perimeter, vertices, and centroid lets you find and mark these values as well as label the shapes