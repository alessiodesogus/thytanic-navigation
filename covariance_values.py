import numpy as np
from vision import get_current_state, init_cam

px = []
py = []
rots = []
cam = init_cam()
for i in range(500):
    _, pos, rot = get_current_state(
        cam=cam,
        obstacle_range=[np.array([0, 150, 25]), np.array([30, 220, 100])],
        target_range=[np.array([10, 80, 245]), np.array([25, 120, 256])],
        th_front_range=[np.array([0, 171, 230]), np.array([15, 245, 256])],
        th_back_range=[np.array([16, 130, 150]), np.array([35, 170, 210])],
    )
    px.append(pos[0])
    py.append(pos[1])
    rots.append(rot)
with open("covariance_test5.txt", "w") as f:
    f.write("xpos = [")
    for pos in px:
        f.write(str(pos))
        f.write(", ")
    f.write("]")

    f.write("ypos = [")
    for pos in py:
        f.write(str(pos))
        f.write(", ")
    f.write("]")

    f.write("orientation = [")
    for rot in rots:
        f.write(str(rot))
        f.write(", ")
    f.write("]")
