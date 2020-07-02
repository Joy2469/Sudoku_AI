# This is a sample Python script.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(img):
    cv2.imshow('Sudoku', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


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


def main():
    print("Enter image name: ")
    image_url = input()
    img = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)

    processed_sudoku = processing(img)
    display_image(processed_sudoku)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# sudoku_1.jpg
