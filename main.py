"""
Aditi Jain
this file calls all other files and is the starting point of the program
"""

from image_prcoesses import extract
from digit_Recognition_CNN import run
from predict import predict



def main():
    # Calling the image_prcoesses.py extract function to get a processed np.array of cells
    image_grid = extract()
    # note we have alreday created and stored the model but if you want to do that again use the following command
    # run()

    # sudoku_grid = predict(image_grid)
    # display_sudoku(sudoku_grid.list())

    # predict
    sudoku = predict(image_grid)



if __name__ == '__main__':
    main()
