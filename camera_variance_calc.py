import numpy as np


def get_list(line: str, el_list: list) -> list[float]:
    """creates a list of floats given a string separated with ,

    Args:
        line (str): _description_
        el_list (list): _description_

    Returns:
        _type_: _description_
    """
    for el in line.split(","):
        if el != "":
            el_list.append(float(el))
    return el_list


x_pos_vars, y_pos_vars, rot_vars = [], [], []
x_pos_stds, y_pos_stds, rot_stds = [], [], []
for i in range(1, 6):
    # reading camera estimation values
    with open(f"camera_variances/camera_values{i}.txt", "r") as f:
        xpos, ypos, rot = [], [], []
        for index, line in enumerate(f):
            line = line.replace("]", "")
            line = line.replace("\n", "")
            line = line.replace(" ", "")
            if index == 0:
                line = line.replace("xpos=[", "")
                get_list(line, xpos)
            if index == 1:
                line = line.replace("ypos=[", "")
                get_list(line, ypos)
            if index == 2:
                line = line.replace("orientation=[", "")
                get_list(line, rot)
    # removing nans from data
    xpos = np.array(xpos)
    xpos = xpos[~np.isnan(xpos)]
    ypos = np.array(ypos)
    ypos = ypos[~np.isnan(ypos)]
    rot = np.array(rot)
    rot = rot[~np.isnan(rot)]
    # variance + standard deviation
    x_pos_vars.append(np.var(xpos))
    x_pos_stds.append(np.std(xpos))
    y_pos_vars.append(np.var(ypos))
    y_pos_stds.append(np.std(ypos))
    rot_vars.append(np.var(rot))
    rot_stds.append(np.std(rot))

print("x positionvariances", np.average(x_pos_vars))
print("y positionvariances", np.average(y_pos_vars))
print("orientation variances", np.average(rot_vars))

"""print("x position standard deviations", x_pos_stds)
print("y position standard deviations", y_pos_stds)
print("orientation standard deviations", rot_stds)"""
