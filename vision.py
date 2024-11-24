import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys
import cv2

# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/


def generate_map(
    image_path: str,
    background_color: np.ndarray,
    obstacle_color: np.ndarray,
    thymio_color: np.ndarray,
    target_color: np.ndarray,
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
        image_path (str): relative path to the image
        background_color (np.ndarray): rgb values for background color
        thymio_color (np.ndarray): rgb values for thymio color
        obstacle_color (np.ndarray): rgb values for obstacle
        target_color (np.ndarray): rgb values for target
    """
    img_arr = take_picture()

    # remove alpha channel if needed
    if np.shape(img_arr)[-1] > 3:
        img_arr = img_arr[:, :, :3]

    map_arr = np.empty((len(img_arr[:, 0]), len(img_arr[0])))

    # looping through the input image and finding the norm for each possible option
    for x in range(len(img_arr[:, 0])):
        for y in range(len(img_arr[0])):
            rgb = img_arr[x, y]
            # hsv color space is more robust against lighting changes when taking 3d norm
            hsv = np.array(colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2]))
            bg_hsv = np.array(
                colorsys.rgb_to_hsv(
                    background_color[0], background_color[1], background_color[2]
                )
            )
            th_hsv = np.array(
                colorsys.rgb_to_hsv(thymio_color[0], thymio_color[1], thymio_color[2])
            )
            obs_hsv = np.array(
                colorsys.rgb_to_hsv(
                    obstacle_color[0], obstacle_color[1], obstacle_color[2]
                )
            )
            tar_hsv = np.array(
                colorsys.rgb_to_hsv(target_color[0], target_color[1], target_color[2])
            )
            bg_norm = np.linalg.norm(hsv - bg_hsv)
            th_norm = np.linalg.norm(hsv - th_hsv)
            obs_norm = np.linalg.norm(hsv - obs_hsv)
            tar_norm = np.linalg.norm(hsv - tar_hsv)
            # assigning the correct map object to the map array
            key = np.argmin(np.array([bg_norm, obs_norm, th_norm, tar_norm]))
            map_arr[x, y] = key

    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam.png", map_arr)
    plt.show()
    return map_arr


def take_picture() -> np.ndarray:
    """takes a picture with the camera of index 1 (if existing), 2 otherwise

    Returns:
        np.ndarray: image taken by the camera in numpy array format
    """
    # https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(2)
    ret = False
    while not ret:
        ret, frame = cam.read()
    return frame


generate_map(
    "inputs/ThymioMapTest.png",
    background_color=np.array([119, 104, 215]),
    obstacle_color=np.array([0, 0, 0]),
    thymio_color=np.array([256, 256, 256]),
    target_color=np.array([256, 0, 0]),
)
