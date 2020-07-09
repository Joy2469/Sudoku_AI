"""
This file asks the user for file name of th sudoku, pre-porcesses it,
finds the corners, crops the sudoku board
and returns the array of the cells of the sudoku
"""

import cv2
import numpy as np


def display_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


# we will blur the image using gaussian blur in order to reduxce the blur in adavaptive thresholding
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a
# gaussian window.
def processing(img, skip_dilate=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Note that kernel sizes must be positive and odd and the kernel must be square.
    process = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, constant(c)) blockSize – Size of
    # a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. C –
    # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be
    # zero or negative as well.
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # we need grid edges so we need to invert colors
    process = cv2.bitwise_not(process, process)

    if not skip_dilate:
        # This is only used for sudoku processing and not for cell processing
        # np.uint8 will wrap.
        # For example, 235+30 = 9.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        process = cv2.dilate(process, kernel)

    return process


def find_corners(img):
    # findContours: boundaries of shapes having same intensity
    # CHAIN_APPROX_SIMPLE - stores only minimal information of points to describe contour
    # -> RETR_EXTERNAL: gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
    # cv2.ContourArea(): Finds area of outermost polygon(largest feature) in img.
    ext_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)

    # loop runs only once
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        # cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
        # Curve-> hers is the largest contour
        # epsilon -> Parameter specifying the approximation accuracy. This is the maximum distance between the original curve and its approximation.
        # closed – If true, the approximated curve is closed. Otherwise, it is not closed.
        # approxPolyDP returns the approximate curve in the same type as the input curve
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            # Here we are looking for the largest 4 sided contour
            return approx

    # Ramer Doughlas Peucker algorithm:
    # bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
    #                                  ext_contours[0]]), key=operator.itemgetter(1))
    # top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
    #                             ext_contours[0]]), key=operator.itemgetter(1))
    # bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
    #                                 ext_contours[0]]), key=operator.itemgetter(1))
    # top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
    #                               ext_contours[0]]), key=operator.itemgetter(1))


def order_corner_points(corners):
    # Corners[0],... stores in format [[x y]]
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return top_l, top_r, bottom_r, bottom_l


# Crop the image
def perspective_transform(image, corners):
    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, grid, (width, height))


def create_image_grid(img):
    grid = np.copy(img)
    # not all sudoku out there have same width and height in the small squares so we need to consider differnt heights and width
    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding the cropped grid and inverting it
    grid = cv2.bitwise_not(grid, grid)


    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("BoardCells1/cell" + str(i) + str(j) + ".jpg")
    except:
        pass
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("BoardCells1/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

    return finalgrid


def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def extract():
    # print("Enter image name: ")
    # image_url = input()
    img = cv2.imread('sudoku_1.jpg')
    processed_sudoku = processing(img)
    sudoku = find_corners(processed_sudoku)
    transformed = perspective_transform(img, sudoku)
    cropped = 'cropped_img.png'
    cv2.imwrite(cropped, transformed)
    transformed = cv2.resize(transformed, (450, 450))
    sudoku = create_image_grid(transformed)
    return sudoku


# if __name__ == '__main__':
#     extract()
