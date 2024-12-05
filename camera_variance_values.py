import numpy as np
from vision import get_current_state, init_cam

px = []
py = []
rots = []
cam = init_cam()
for i in range(500):
    _, pos, rot = get_current_state(
        cam=cam,
        obstacle_range=[np.array([0, 130, 0]), np.array([160, 240, 140])],
        target_range=[np.array([0, 30, 235]), np.array([255, 129, 256])],
        th_front_range=[np.array([0, 130, 253]), np.array([15, 250, 256])],
        th_back_range=[np.array([16, 130, 141]), np.array([40, 250, 252])],
    )
    px.append(pos[0])
    py.append(pos[1])
    rots.append(rot)
with open("camera_variances/camera_values5.txt", "w") as f:
    f.write("xpos = [")
    for pos in px:
        f.write(str(pos))
        f.write(", ")
    f.write("] \n")

    f.write("ypos = [")
    for pos in py:
        f.write(str(pos))
        f.write(", ")
    f.write("] \n")

    f.write("orientation = [")
    for rot in rots:
        f.write(str(rot))
        f.write(", ")
    f.write("] \n")
