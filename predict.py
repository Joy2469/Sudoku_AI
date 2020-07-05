"""
This class takes img_grid
"""
import keras
from keras.models import load_model
import numpy as np
import cv2


def preprocess_image(img):
    rows = np.shape(img)[0]

    # First we need to remove the outermost white pixels.
    # This can be achieved by flood filling with some of the outer points as seeds.
    # After looking at the cell images, I concluded that it's enough if we
    # Flood fill with all the points from the three outermost layers as seeds
    for i in range(rows):
        # Floodfilling the outermost layer
        cv2.floodFill(img, None, (0, i), 0)
        cv2.floodFill(img, None, (i, 0), 0)
        cv2.floodFill(img, None, (rows - 1, i), 0)
        cv2.floodFill(img, None, (i, rows - 1), 0)
        # Floodfilling the second outermost layer
        cv2.floodFill(img, None, (1, i), 1)
        cv2.floodFill(img, None, (i, 1), 1)
        cv2.floodFill(img, None, (rows - 2, i), 1)
        cv2.floodFill(img, None, (i, rows - 2), 1)


    cv2.imwrite("StagesImages/14.jpg", img)
    # Finding the bounding box of the number in the cell
    rowtop = None
    rowbottom = None
    colleft = None
    colright = None
    thresholdBottom = 50
    thresholdTop = 50
    thresholdLeft = 50
    thresholdRight = 50
    center = rows // 2
    for i in range(center, rows):
        if rowbottom is None:
            temp = img[i]
            if np.sum(temp) < thresholdBottom or (i == rows - 1):
                rowbottom = i
        if rowtop is None:
            temp = img[rows - i - 1]
            if np.sum(temp) < thresholdTop or i == rows - 1:
                rowtop = rows - i - 1
        if colright is None:
            temp = img[:, i]
            if np.sum(temp) < thresholdRight or i == rows - 1:
                colright = i
        if colleft is None:
            temp = img[:, rows - i - 1]
            if np.sum(temp) < thresholdLeft or i == rows - 1:
                colleft = rows - i - 1

    # Centering the bounding box's contents
    newimg = np.zeros(np.shape(img))
    startatX = (rows + colleft - colright) // 2
    startatY = (rows - rowbottom + rowtop) // 2
    for y in range(startatY, (rows + rowbottom - rowtop) // 2):
        for x in range(startatX, (rows - colleft + colright) // 2):
            newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]


    cv2.imwrite("StagesImages/15.jpg", newimg)
    return newimg

def predict():
    # sudoku_str = [[0 for i in range(9)] for j in range(9)]
    # threshold = 5*255

    # for i in range(9):
    #     for j in range(9):
    #         img = np.copy(img_grid[i][j])
    #         img = preprocess_image(img)
    #         cv2.imwrite("CleanedBoardCells/cell" + str(i) + str(j) + ".jpg", img)
    #         img = cv2.resize(img, (28, 28))
    #         model = keras.loadmodel('cnn.hdf5')
    #         arr = np.array(img).reshape((28, 28, 3))
    #         arr = np.expand_dims(arr, axis = 0)
    #         prediction = model.predict(arr)[0]
    #         bestclass =''
    #         bestconf =-1
    #         for n in range(9):
    #             if(prediction[n]>bestconf):
    #                 bestclass = str(n)
    #                 bestconf = prediction[n]
    #         print("I think didgit is ", bestclass, " with ", str(bestconf*100), " % confidence")
    #         sudoku_str[i][j] = bestclass

    img = cv2.imread('no5.jpg')
    img = np.copy(img)
    img = preprocess_image(img)
    cv2.imwrite("CleanedBoardCells/cell.jpg", img)
    img = cv2.resize(img, (28, 28))
    model = load_model('cnn.hdf5')
    arr = np.array(img).reshape((img.shape[0], 28, 28, 1))
    arr = np.expand_dims(arr, axis = 0)
    prediction = model.predict(arr)[0]
    bestclass =''
    bestconf =-1
    for n in range(9):
        if(prediction[n]>bestconf):
            bestclass = str(n)
            bestconf = prediction[n]
    print("I think didgit is ", bestclass, " with ", str(bestconf*100), " % confidence")
    sudoku_str = bestclass

    return bestclass

predict()

