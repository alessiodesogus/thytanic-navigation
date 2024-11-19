import numpy as np, math, scipy


def downsampling(image: np.ndarray, endsize: list, obstacledilation: int, threshold: float):
    """
    this function downsamples a binary numpy 2darray into the endsize dimensions
    1= obstacle pixel and 0 free ground pixel
    it does so by linearly separating the image into groups, averaging the value of each group
    and if the group's proportion of obstacle pixels is above the threshold, it classifies that new pixel as obstacle

    obstacledilation dilates the obstacles with a nxn filter pixels before performing any operation, leave to 0 for no dilation
    the function assumes that end size is at most half the size of the starter size

    """
    startsize = list(np.shape(image))
    print(image)
    print(startsize)
    print(endsize)

    if startsize <= endsize:
        print("the goal is larger than the image, check again")
        print("goal dimensions: " + str(endsize) + ". image dimensions: " + str(startsize))
        return -1
    # dilation function
    if obstacledilation != 0:
        image = scipy.ndimage.binary_dilation(image, np.ones([obstacledilation, obstacledilation], int)).astype(int)


    windowsize = [math.ceil(startsize[0] / endsize[0]), math.ceil(startsize[1] / endsize[1])]
    endimage = np.zeros(endsize, int)


    for k in range(endsize[0]):
        for j in range(endsize[1]):
            # runs the window, selects a submatrix and caps to the sides, we do not pad
            window = image[k * windowsize[0]:min((k + 1) * windowsize[0], startsize[0]), :][:,
                     j * windowsize[1]:min((j + 1) * windowsize[1], startsize[1])]
            avg = np.sum(window) / window.size
            if avg > threshold:
                endimage[k, j] = 1

    return endimage

def pathfinder(startpoint, endpoint, area):
    xarea, yarea = np.shape(area))
    unexplored = np.asarray([startpoint + [0, -1, -1]])
    unexplored = np.append(unexplored, [startpoint + [3, -1, -1]], axis = 0)

    print(unexplored)
    while unexplored.any():
        #select point wiht the best f-value, and remove it from unexplored
        currentid = np.argmin(unexplored[:,2])
        current = unexplored[currentid]
        np.delete(unexplored, currentid)
        print(current[0:1])
        if current[0:1]:
            #smashy return lines TODO
            return current
        children = [[current[0]-1, current[1]],[current[0], current[1]+1],[current[0]+1, current[1]],[current[0], current[1]-1]]
        for child in children:
            #detect obstacle or out of bounds
            if not (0 <= child[0] < xarea and 0 <= child[1] < yarea) or area[child[0], child[1]] == 1:
                children.remove(child)
            else:
                return 0
        break





# example downsampling input
# downsampling(np.asarray(7 * [[0, 1, 0, 0, 1, 1, 0]]), [3, 3], 5, 0.6)
pathfinder([0,0], [5,0], np.asarray(7 * [[0, 0, 0, 0, 1, 1, 0]]))