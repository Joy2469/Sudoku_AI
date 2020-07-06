"""
Predict an image
accepts image grid
return preidcted grid
"""
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import ImageOps
from image_prcoesses import extract, scale_and_centre


def display_image(img):
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all windows


def pre_pro(img):


    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, constant(c)) blockSize – Size of
    # a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. C –
    # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be
    # zero or negative as well.
    process = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 2)

    # we need grid edges so we need to invert colors
    img = cv2.bitwise_not(process, process)
    display_image(img)

    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    scan_tl = None
    scan_br = None

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

        # Colour everything grey (compensates for features outside of our middle scanning range
        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 255 and x < width and y < height:
                    cv2.floodFill(img, None, (x, y), 64)

        mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

        # Highlight the main feature
        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, mask, seed_point, 255)

        top, bottom, left, right = height, 0, width, 0

        for x in range(width):
            for y in range(height):
                if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                    cv2.floodFill(img, mask, (x, y), 0)

                # Find the bounding parameters
                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

        bbox = [[left, top], [right, bottom]]
    display_image(img)
    find_corners(img)

def find_corners():
    #getting image grid
    img_grid = extract()
    sudoku =[]
    for i in range(9):
        for j in range(9):

            gray = img_grid[i][j]
            gray = cv2.resize(gray, (28, 28))
            original = gray.copy()
            thresh = 128 # define a threshold, 128 is the middle of black and white in grey scale
            # threshold the image
            gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

            # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # canny = cv2.Canny(blurred, 120, 255, 1)
            # kernel = np.ones((5, 5), np.uint8)
            # dilate = cv2.dilate(canny, kernel, iterations=1)

            # Find contours
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            #Note the number is always placed in the center
            # Since image is 28x28

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if (x < 3 or y < 3 or h < 3 or w < 3):
                    continue
                ROI = gray[y:y + h, x:x + w]

                cv2.imwrite("CleanedBoardCells/cell{}{}.png".format(i, j), ROI)

    # for i in range(9):
    #     for j in range(9):
    #         cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), )


def predict():

    image = scale_and_centre(cv2.imread('CleanedBoardCells/cell04.png'), 28)
    display_image(image)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)

    model = load_model('cnn.hdf5')
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

    print(pred.argmax())


predict()
