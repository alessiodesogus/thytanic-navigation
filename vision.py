import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys
import cv2
from scipy import ndimage

# https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/


def generate_map(
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
        background_color (np.ndarray): rgb values for background color
        thymio_color (np.ndarray): rgb values for thymio color
        obstacle_color (np.ndarray): rgb values for obstacle
        target_color (np.ndarray): rgb values for target
    """
    if img_path == "":
        img_arr = take_picture()
    else:
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_arr)
    plt.colorbar()
    plt.imsave("output/picture2.png", img_arr)
    plt.show()
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
            """bg_norm = np.linalg.norm(rgb - background_color)
            th_norm = np.linalg.norm(rgb - thymio_color)
            obs_norm = np.linalg.norm(rgb - obstacle_color)
            tar_norm = np.linalg.norm(rgb - target_color)"""
            # assigning the correct map object to the map array
            key = np.argmin(np.array([bg_norm, obs_norm, th_norm, tar_norm]))
            map_arr[x, y] = key
    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam.png", map_arr)
    plt.show()
    # https://stackoverflow.com/questions/48013355/eliminating-number-of-connected-pixels-smaller-than-some-specified-number-thresh
    # remove noise in the output map
    labels, Nlabels = ndimage.measurements.label(map_arr)
    label_size = [(labels == label).sum() for label in range(Nlabels + 1)]
    for label, size in enumerate(label_size):
        print("label %s is %s pixels in size" % (label, size))

    # now remove the labels
    for label, size in enumerate(label_size):
        if size < 500:
            map_arr[labels == label] = 0
    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map_hsv_cam_noise_removed.png", map_arr)
    plt.show()
    return map_arr


def take_picture() -> np.ndarray:
    """takes a picture with the camera of index 1 (if existing), 2 otherwise

    Returns:
        np.ndarray: image taken by the camera in numpy array format
    """
    # https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
    ret = False
    while not ret:
        ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


generate_map(
    background_color=np.array([119, 104, 215]),
    obstacle_color=np.array([0, 0, 0]),
    thymio_color=np.array([255, 190, 130]),
    target_color=np.array([255, 140, 100]),
)
