# How to Run Code

Dependencies:

```
pip install opencv-python numpy scipy
```

Part 1 takes a still image and writes `part1_detected.png` and `part1_mask.png`
next to it. It has no options.

```
python part1.py PennAir_2024_App_Static.png
```

Parts 2, 3 and 4 take a video and all share the same format:

```
python part2.py PennAir_2024_App_Dynamic.mp4
python part3.py PennAir_2024_App_Dynamic_Hard.mp4
python part4.py PennAir_2024_App_Dynamic_Hard.mp4
```

A window opens showing the result. Press `q` to quit.

Add

```
-o "output file"
```

to have the program write the video produced somewhere. The other options are

```
--no-show     don't open a window (needed over SSH, and faster)
-s 0.4        detection scale; lower is faster and less accurate
--limit 300   stop after N frames
--debug       inset the binary mask in the corner (parts 3 and 4)
```

Part 4 additionally takes `--radius` for the real circle radius in inches
(default 10) and `--centre-principal-point`, which moves the principal point
to the middle of the image instead of the corner the given intrinsic matrix
specifies. See the part 4 reflection for why that matters.

Part 4 imports `part3.py`, so the two files have to sit in the same folder.

ROS instructions are in the readme in the ros2_ws folder.
