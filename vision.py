import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    img = Image.open(image_path)
    img_arr = np.asarray(img)
    # remove alpha channel if needed
    if np.shape(img_arr)[-1] > 3:
        img_arr = img_arr[:, :, :3]

    map_arr = np.empty((len(img_arr[:, 0]), len(img_arr[0])))

    for x in range(len(img_arr[:, 0])):
        for y in range(len(img_arr[0])):
            rgb = img_arr[x, y]
            bg_norm = np.linalg.norm(rgb - background_color)
            th_norm = np.linalg.norm(rgb - thymio_color)
            obs_norm = np.linalg.norm(rgb - obstacle_color)
            tar_norm = np.linalg.norm(rgb - target_color)
            key = np.argmin(np.array([bg_norm, obs_norm, th_norm, tar_norm]))
            map_arr[x, y] = key

    # display map
    plt.imshow(map_arr)
    plt.colorbar()
    plt.imsave("output/map.png", map_arr)
    plt.show()
    return map_arr


generate_map(
    "inputs/ThymioMapTest.png",
    background_color=np.array([119, 104, 215]),
    obstacle_color=np.array([0, 0, 0]),
    thymio_color=np.array([256, 256, 256]),
    target_color=np.array([256, 0, 0]),
)
