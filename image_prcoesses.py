# This is a sample Python script.
import operator

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(img):
    cv2.imshow('Sudoku', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    display_image(img)
    return img


# we will blur the image using gaussian blur in order to reduxce the blur in adavaptive thresholding
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a
# gaussian window.
def processing(img):
    # Note that kernel sizes must be positive and odd and the kernel must be square.
    process = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, constant(c)) blockSize – Size of
    # a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. C –
    # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be
    # zero or negative as well.
    process = cv2.adaptiveThreshold(process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # we need grid edges so we need to invert colors
    process = cv2.bitwise_not(process, process)
    # np.uint8 will wrap.
    # For example, 235+30 = 9.
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    process = cv2.dilate(process, kernel)

    return process


def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

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

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def get_corners(img, original):
    # findContours: boundaries of shapes having same intensity
    # CHAIN_APPROX_SIMPLE - stores only minimal information of points to describe contour
    # -> RETR_EXTERNAL: gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
    # cv2.ContourArea(): Finds area of outermost polygon(largest feature) in img.
    ext_contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1]
    ext_contours = sorted(ext_contours, key=cv2.contourArea, reverse=True)
    # Sort by area, descending
    # Therefore the largest ploygon is stored in contours[0]

    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        transformed = perspective_transform(original, approx)
        break


def main():
    print("Enter image name: ")
    image_url = input()
    img = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)

    processed_sudoku = processing(img)
    get_corners(processed_sudoku, image_url)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# sudoku_1.jpg
