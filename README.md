# Sudoku_openCV
We will be creating a Sudoku Solver AI using python and Open CV to read a Sudoku puzzle from an image and solving it. There a lot of methods to achieve this goal. Thus in this series, I have compiled the best methods I could find/research along with some hacks/tricks I learned along the way.

## Tutorial 
Access the written tutorial of [SudkoAI](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629). This article is a part of the series Sudoku Solver AI with OpenCV. \
**Part 1:** [Image Processing](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629) \
**Part 2:** [Sudoku and Cell Extraction](https://medium.com/@aditijain0424/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066) \
**Part 3:** [Solving the Sudoku](https://medium.com/@aditijain0424/part-3-solving-the-sudoku-ai-solver-13f64a090922)
## Steps
1. **Import the image**
2. **Pre Processing the Image** \
   2.1 Gaussian blur: We need to gaussian blur the image to reduce noise in thresholding algorithm \
   2.2 Thresholding: Segmenting the regions of the image \
   2.3 Dilating the image: In cases like noise removal, erosion is followed by dilation.
3. **Sudoku Extraction** \
3.1 Find Contours \
3.2 Find Corners: Using Ramer Doughlas Peucker algorithm / approxPolyDP for finding corners \
3.3 Crop and Warp Image: We remove all the elements in the image except the sudoku \
3.4 Extract Cells 
4. **Interpreting the Digits** \
4.1 Import the libraries and load the dataset \
4.2 Preprocess the data \
4.3 Creating the Model \
4.4 Predicting the digits
5. **Solving the Sudoku**

## Steps in deatil
###
