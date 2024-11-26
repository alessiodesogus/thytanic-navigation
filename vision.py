import numpy as np
import matplotlib.pyplot as plt
import colorsys
import cv2
from scipy import ndimage

# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/


def get_current_state(
    cam: cv2.VideoCapture,
    background_color: np.ndarray,
    obstacle_color: np.ndarray,
    thymio_color: np.ndarray,
    target_color: np.ndarray,
    img_path: str = "",
) -> np.ndarray:
    """Function that takes the path to an image of the map that contains the Thymio and the Target.
        It then separates the image based on color between background, thymio, obstacles, and target.
        the separation is done by using the euclidian norm in the rgb space.
        It returns a numpy array with the following entries:
            0 for background
            1 for obstacle
            2 for thymio
            3 for target

    Args:
        cam (cv2.VideoCapture): connected camera
        background_color (np.ndarray): rgb values for background color
        thymio_color (np.ndarray): rgb values for thymio color
        obstacle_color (np.ndarray): rgb values for obstacle
        target_color (np.ndarray): rgb values for target
        img_path (str): if this argument is equal to "", a new picture is taken by the supplied camera, otherwise the image is loaded from img_path
    """
    if img_path == "":
        img_arr = take_picture(cam)
    else:
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_arr)
    plt.colorbar()
    plt.imsave("output/picture2.png", img_arr)
    plt.show()

    orientation = get_orientation(cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY))
    print(orientation)
    # reduce amount of pixels in image to speed up processing
    print(np.size(img_arr))
    img_arr = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img_arr)))
    print(np.size(img_arr))
    # remove alpha channel if needed
    if np.shape(img_arr)[-1] > 3:
        img_arr = img_arr[:, :, :3]

    map_arr = np.empty((len(img_arr[:, 0]), len(img_arr[0])))
    obstacles = np.zeros_like(map_arr)
    # converting colors to hsv space
    bg_hsv = np.array(
        colorsys.rgb_to_hsv(
            background_color[0], background_color[1], background_color[2]
        )
    )
    th_hsv = np.array(
        colorsys.rgb_to_hsv(thymio_color[0], thymio_color[1], thymio_color[2])
    )
    obs_hsv = np.array(
        colorsys.rgb_to_hsv(obstacle_color[0], obstacle_color[1], obstacle_color[2])
    )
    tar_hsv = np.array(
        colorsys.rgb_to_hsv(target_color[0], target_color[1], target_color[2])
    )
    # looping through the input image and finding the norm for each possible option
    for x in range(len(img_arr[:, 0])):
        for y in range(len(img_arr[0])):
            rgb = img_arr[x, y]
            # hsv color space is more robust against lighting changes when taking 3d norm
            hsv = np.array(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
            bg_norm = np.linalg.norm(hsv - bg_hsv)
            th_norm = np.linalg.norm(hsv - th_hsv)
            obs_norm = np.linalg.norm(hsv - obs_hsv)
            tar_norm = np.linalg.norm(hsv - tar_hsv)

            # assigning the correct map object to the map array
            key = np.argmin(np.array([bg_norm, obs_norm, th_norm, tar_norm]))
            map_arr[x, y] = key
            if key == 1:
                obstacles[x, y] = 1
    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam.png", map_arr)
    plt.show()
    # https://stackoverflow.com/questions/48013355/eliminating-number-of-connected-pixels-smaller-than-some-specified-number-thresh
    # remove noise in the output map
    labels, nlabels = ndimage.measurements.label(obstacles)
    label_size = [(labels == label).sum() for label in range(nlabels + 1)]

    # now remove the labels
    for label, size in enumerate(label_size):
        if size < 50:
            map_arr[labels == label] = 0
    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam_noise_removed.png", map_arr)
    plt.show()
    return map_arr, orientation


def take_picture(cam: cv2.VideoCapture) -> np.ndarray:
    """takes a picture with the camera supplied as an argument

    Returns:
        np.ndarray: image taken by the camera in numpy array format in rgb channels
    """
    ret = False
    while not ret:
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
    return cam


def get_orientation(img: np.ndarray) -> float:
    """searches the suplied image for a qr code. if a code is found it returns the orientation of the code relative to the camera

    Args:
        img (np.ndarray): grayscale image of the camera

    Returns:
        float: orientation of the qr code relative to the image in rad, -1000 if no qr code was found
    """
    # https://temugeb.github.io/python/computer_vision/2021/06/15/QR-Code_Orientation.html
    qr = cv2.QRCodeDetector()
    ret_qr, points = qr.detect(img)
    print(ret_qr)
    if ret_qr:
        # get angle between points in cam coordinate frame
        p0 = points[0][0]
        p1 = points[0][1]
        orientation = np.atan((p1[1] - p0[1]) / (p1[0] - p0[0]))
        print(np.rad2deg(orientation))
        return orientation
    return -1000


"""cam = init_cam()
img = take_picture(cam)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
get_orientation(img)
plt.imshow(img)
plt.colorbar()
plt.imsave("output/qrtest.png", img)
plt.show()"""
