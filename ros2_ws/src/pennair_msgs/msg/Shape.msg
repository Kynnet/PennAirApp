# One detected shape, as published by the shape_detector node.

int32   id              # track id, stable while the shape is followed
string  label           # "circle", "triangle", ... or "merged shapes"
int32   edges           # vertices in the outline; 0 for a circle
bool    overlapped      # another shape is on top of this one
bool    clipped         # shape is partly outside the frame

geometry_msgs/Point centroid    # pixels, full resolution; z unused
geometry_msgs/Point[] outline   # polygon vertices in pixels; z unused

geometry_msgs/Point position    # camera frame, inches (part 4)
bool has_position               # false until the circle has set the scale
