import numpy as np, math, scipy


def downsampling(
    image: np.ndarray, endsize: list, obstacledilation: int, threshold: float
):
    """
    this function downsamples a binary numpy 2darray into the endsize dimensions
    1= obstacle pixel and 0 free ground pixel
    it does so by linearly separating the image into groups, averaging the value of each group
    and if the group's proportion of obstacle pixels is above the threshold, it classifies that new pixel as obstacle

    obstacledilation dilates the obstacles with a nxn filter pixels before performing any operation, leave to 0 for no dilation
    the function assumes that end size is at most half the size of the starter size

    """
    startsize = list(np.shape(image))

    if startsize < endsize:
        print("the goal is larger than the image, check again")
        print(
            "goal dimensions: " + str(endsize) + ". image dimensions: " + str(startsize)
        )
        return -1
    # dilation function
    if obstacledilation != 0:
        image = scipy.ndimage.binary_dilation(
            image, np.ones([obstacledilation, obstacledilation], int)
        ).astype(int)

    windowsize = [
        math.ceil(startsize[0] / endsize[0]),
        math.ceil(startsize[1] / endsize[1]),
    ]
    endimage = np.zeros(endsize, int)

    for k in range(endsize[0]):
        for j in range(endsize[1]):
            # runs the window, selects a submatrix and caps to the sides, we do not pad
            window = image[
                k * windowsize[0] : min((k + 1) * windowsize[0], startsize[0]), :
            ][:, j * windowsize[1] : min((j + 1) * windowsize[1], startsize[1])]

            avg = np.sum(window) / window.size
            if avg > threshold:
                endimage[k, j] = 1

    return endimage


def pathmaker(point, history):
    path = [point[0:2]]  # Initialize path with the endpoint
    k = 0
    while k < 1000:  # Prevent infinite loops in case of unexpected errors
        k += 1
        # Check if the point has no parent (-1, -1), indicating the start point
        if np.array_equal(point[4:6], [-1, -1]):
            return np.array(
                path[::-1]
            )  # Return the reversed path (from start to endpoint)
        else:
            # Find the parent of the current point in the history
            parent_indices = np.where((history[:, 0:2] == point[4:6]).all(axis=1))[0]
            if len(parent_indices) == 0:
                raise ValueError(
                    "Parent point not found in history! Path reconstruction failed."
                )

            parent_index = parent_indices[0]
            point = history[parent_index]  # Move to the parent point
            path.append(point[0:2])  # Append the parent's coordinates to the path


def pathfinder(startpoint, endpoint, area):
    # pathfinder takes in a start and an endpoint, which must be 2d coordinates [x, y] that fit within area
    #area must be a binary 2d matrix where 1 is an obstacle pixel, and 0 a free path pixel
    xarea, yarea = np.shape(area)
    # A point is composed of x, y, path length, heuristic, and parent point x and y
    unexplored = np.array(
        [
            startpoint
            + [0, sum((x - y) ** 2 for x, y in zip(startpoint, endpoint)), -1, -1]
        ],
        int,
    )
    explored = np.zeros((0, 6), int)  # Initialize explored as an empty 2D array

    while len(unexplored) > 0:  # Ensure unexplored is non-empty

        # Select point with the best f-value and remove it from unexplored
        currentid = np.argmin(unexplored[:, 3])
        current = unexplored[currentid]
        explored = np.vstack([explored, current])  # Add to explored
        unexplored = np.delete(unexplored, currentid, axis=0)  # Remove selected point

        # If the current point is the endpoint, return the path
        if np.array_equal(current[0:2], endpoint):

            return (
                pathmaker(current, explored),
                explored,
                unexplored,
            )  # Adjust this to return the desired path format

        # Generate the four potential children
        children = [
            np.array(
                [current[0] - 1, current[1], current[2] + 1, 0, current[0], current[1]]
            ),
            np.array(
                [current[0], current[1] + 1, current[2] + 1, 0, current[0], current[1]]
            ),
            np.array(
                [current[0] + 1, current[1], current[2] + 1, 0, current[0], current[1]]
            ),
            np.array(
                [current[0], current[1] - 1, current[2] + 1, 0, current[0], current[1]]
            ),
        ]

        for child in children:
            # Detect if child is out of bounds or an obstacle
            if (
                0 <= child[0] < xarea
                and 0 <= child[1] < yarea
                and area[int(child[0]), int(child[1])] != 1
            ):
                # Check if child is already in unexplored or explored
                in_unexplored = (
                    any((child[0:2] == unexplored[:, 0:2]).all(axis=1))
                    if len(unexplored) > 0
                    else False
                )
                in_explored = (
                    any((child[0:2] == explored[:, 0:2]).all(axis=1))
                    if len(explored) > 0
                    else False
                )

                if not in_unexplored and not in_explored:
                    # Calculate heuristic: Euclidean distance to endpoint + path length
                    child[3] = (
                        sum((x - y) ** 2 for x, y in zip(child[0:2], endpoint))
                        + child[2]
                    )

                    unexplored = np.vstack(
                        [unexplored, child]
                    )  # Add child to unexplored

    print("No path found.")
    return None


def downsamplingprep(image: np.ndarray, endsize: list, dilation: int, erosion: int):
    outputimage = (image == 1).astype(int)  # obstacles only channel
    thymage = (image == 2).astype(int)  # thymio channel
    gimage = (image == 3).astype(int)  # end goal channel

    if erosion != 0:
        thymage = scipy.ndimage.binary_erosion(
            thymage, np.ones([erosion, erosion], int)
        ).astype(int)
        gimage = scipy.ndimage.binary_erosion(
            gimage, np.ones([erosion, erosion], int)
        ).astype(int)

    if dilation != 0:
        thymage = scipy.ndimage.binary_dilation(
            thymage, np.ones([erosion, erosion], int)
        ).astype(int)
        gimage = scipy.ndimage.binary_dilation(
            gimage, np.ones([erosion, erosion], int)
        ).astype(int)

    tx, ty = np.where(thymage == 1)
    ex, ey = np.where(gimage == 1)

    thymiopos = [np.average(tx), np.average(ty)]
    endpos = [np.average(ex), np.average(ey)]

    startsize = list(np.shape(image))

    divsize = [
        math.ceil(startsize[0] / endsize[0]),
        math.ceil(startsize[1] / endsize[1]),
    ]

    thymiopos = [
        math.ceil(thymiopos[0] / divsize[0]),
        math.ceil(thymiopos[1] / divsize[1]),
    ]
    endpos = [math.ceil(endpos[0] / divsize[0]), math.ceil(endpos[1] / divsize[1])]

    return thymiopos, endpos, outputimage


# example downsampling input
# downsampling(np.asarray(7 * [[0, 1, 0, 0, 1, 1, 0]]), [3, 3], 5, 0.6)
# print(pathfinder([0, 0], [5, 0], np.asarray(7 * [[0, 0, 0, 0, 1, 1, 0]])))
