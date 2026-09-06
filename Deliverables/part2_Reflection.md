# Approach

I attempted to keep part 1's idea intact and extends it from a still image to video. The local standard deviation of a kernel still separates the smooth shapes from the noisy grass, and the binary bitmask still comes from thresholding it. 

However, that didn't work as well as I wanted, so two features were added.

The first is separating shapes that touch. The split is done based on color. the blob is converted to LAB, quantised into coarse bins, and each occupied bin is taken as one shape. A blob is only split if its interior color actually varies, measured after eroding away the anti-aliased rim, so an ordinary single shape is never cut in half by its own soft edges. After splitting, a solidity test throws out anything that is not convex.

The second is identity. Each shape needs to keep the same label and the same identity from frame to frame, so a tracker assigns detections to existing tracks by solving one global assignment rather than greedily taking the nearest match. The cost combines how far a detection is from a track and how different their mean LAB colors are, with both terms normalised so they are comparable and each given a hard limit beyond which the pairing is refused outright. Color is the more discriminating of the two here, since the shapes are flat and vividly different from one another, so it carries the larger weight. The name of a shape is then a majority vote over a window of recent frames rather than whatever this frame happened to measure.

# Challenges

Shapes that touch were the main problem, and the color split is the answer to it, but it introduces a failure of its own: a single shape whose edge pixels blend into the grass has a wide spread of colors near its rim, which looks exactly like two shapes meeting. Eroding the blob before measuring the spread, and only splitting when the variation exceeds a tolerance, keeps a lone shape whole.

Shapes leaving and entering the frame cause two separate problems. Their area drops below the minimum that filters out noise, so the threshold is relaxed for anything touching the border, and a shape sliding out of frame is not silently dropped. Their vertex count is also wrong while they are clipped, because the frame edge cuts a straight line across them and a triangle briefly reads as a quadrilateral. Detections touching the border are therefore allowed to keep their track alive but not to vote on its name.

Grass occasionally produces a patch that is flat and convex enough to pass every test for a single frame. Rather than tightening the thresholds until real shapes start being lost, a track has to be seen several times before it is reported at all. 

The opposite mistake is drawing a shape that is no longer there. A track that fails to match is kept alive internally for a while, so that it can reclaim its identity if the shape reappears, but it is not reported while unmatched.

Occlusion also disturbs the color that identifies a track. When a shape is partly covered its measured mean color shifts toward whatever is covering it, and if that were stored directly the track would drift away from its true appearance and stop matching itself. The stored color is therefore smoothed over time rather than replaced, so a single bad frame moves it only slightly.

Finally, the label itself flickers. The vertex count from approxPolyDP changes by one now and then as the contour wobbles, so a pentagon reads as a hexagon for a frame. Voting over a window of recent frames removes almost all of that, at the cost of the name taking a few frames to settle when a shape first appears.
