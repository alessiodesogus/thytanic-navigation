import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import time

# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/


def get_current_state(
    cam: cv2.VideoCapture,
    obstacle_range: list[np.ndarray],
    target_range: list[np.ndarray],
    th_back_range: list[np.ndarray],
    th_front_range: list[np.ndarray],
    img_path: str = "",
) -> tuple[np.ndarray, float]:
    """Function that takes the path to an image of the map that contains the Thymio and the Target.
        It then separates the image based on color between background, thymio, obstacles, and target.
        the separation is done by using the euclidian norm in the rgb space.
        It returns a numpy array with the following entries:
            0 for background
            1 for obstacle
            2 for thymio
            3 for target
        And the position and orientation of the Thymio in pixels/rad

    Args:
        cam (cv2.VideoCapture): connected camera
        obstacle_range (list[np.ndarray]): range of allowed hsv values for obstacles
        target_range (list[np.ndarray]): hsv range for target
        th_back_range (list[np.ndarray]): range of allowed hsv values for the back of the thymio
        th_front_image (list[np.ndarray]): hsv range for the front of the thymio
        img_path (str): if this argument is equal to "", a new picture is taken by the supplied camera, otherwise the image is loaded from img_path
    """
    if img_path == "":
        img_arr = take_picture(cam)
    else:
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    # reduce amount of pixels in image to speed up processing
    img_arr = cv2.pyrDown(cv2.pyrDown(img_arr))
    # hsv color space is more robust against lighting changes when taking 3d norm
    img_arr_hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
    map_arr = np.empty((len(img_arr[:, 0]), len(img_arr[0])))
    obstacles = np.zeros_like(map_arr)
    back_image = np.zeros_like(map_arr)
    front_image = np.zeros_like(map_arr)
    # print(obstacle_range)
    # looping through the input image and checking for each pixel if it lies in one of the ranges
    for x in range(len(img_arr[:, 0])):
        for y in range(len(img_arr[0])):
            hsv = img_arr_hsv[x, y]
            key = 0
            if in_hsv_range(hsv, obstacle_range):
                key = 1
            if in_hsv_range(hsv, th_back_range):
                key = 2
                back_image[x, y] = 1
            if in_hsv_range(hsv, th_front_range):
                key = 2
                front_image[x, y] = 1
            if in_hsv_range(hsv, target_range):
                key = 3
            # assigning the correct map object to the map array
            map_arr[x, y] = key
            if key == 1:
                obstacles[x, y] = 1

    tx, ty = np.where(back_image == 1)
    back_pos = [np.average(tx), np.average(ty)]
    # print("position of front of the thymio")
    # print(back_pos)

    tx, ty = np.where(front_image == 1)
    front_pos = [np.average(tx), np.average(ty)]
    # print("position of back of the thymio")
    # print(front_pos)
    orientation = get_orientation(back_pos, front_pos)

    tx, ty = np.where(map_arr == 2)
    thymio_pos = [np.average(tx), np.average(ty)]
    # https://stackoverflow.com/questions/48013355/eliminating-number-of-connected-pixels-smaller-than-some-specified-number-thresh
    # remove noise in the output map
    labels, nlabels = ndimage.measurements.label(obstacles)
    label_size = [(labels == label).sum() for label in range(nlabels + 1)]

    # now remove the labels
    for label, size in enumerate(label_size):
        if size < 25:
            map_arr[labels == label] = 0
    # display map
    """plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam_noise_removed.png", map_arr)
    plt.show()"""
    return (
        map_arr,
        thymio_pos,
        orientation - np.pi / 2,
    )


def take_picture(cam: cv2.VideoCapture) -> np.ndarray:
    """takes a picture with the camera supplied as an argument

    Returns:
        np.ndarray: image taken by the camera in numpy array format in rgb channels
    """
    ret = False
    while not ret:
        time.sleep(0.01)
        ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def init_cam() -> cv2.VideoCapture:
    """returns camera of index 1 or 2, should be the right camera but might als be the webcam depending on pc

    Returns:
        cv2.VideoCapture: camera video capture of the connected camera
    """
    # https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    time.sleep(1)
    return cam


def get_orientation(p0: list[np.float64], p1: list[np.float64]) -> float:
    """searches the suplied image for a qr code. if a code is found it returns the orientation of the code relative to the camera

    Args:
        img (np.ndarray): grayscale image of the camera

    Returns:
        float: orientation of the qr code relative to the image in rad
    """
    orientation = np.atan2((p1[1] - p0[1]), (p1[0] - p0[0]))
    return orientation


def in_hsv_range(hsv: np.ndarray, hsv_range: list[np.ndarray]) -> bool:
    """Given hsv values for a given pixel checks if it lies in a predefined hsv range

    Args:
        hsv (np.ndarray): hsv values of a pixel
        hsv_range (list[np.ndarray]): range of hsv values (min, max) of the desired object

    Returns:
        bool: True if hsv values lie in range, false otherwise
    """
    for i, _ in enumerate(hsv):
        if not (hsv[i] > hsv_range[0][i] and hsv[i] < hsv_range[1][i]):
            return False
    return True


def tune_hsv(img: np.ndarray):
    """function only used to get the correct hsv values

    Args:
        img (np.ndarray): rgb camera image of the scene
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    plt.title("h")
    plt.imshow(img_hsv[:, :, 0])
    plt.colorbar()
    plt.show()
    plt.title("s")
    plt.imshow(img_hsv[:, :, 1])
    plt.colorbar()
    plt.show()
    plt.title("v")
    plt.imshow(img_hsv[:, :, 2])
    plt.colorbar()
    plt.show()


"""cam = init_cam()
img = take_picture(cam)
for i in range(5):
    tune_hsv(img)"""
