# Sudoku_openCV
We will be creating a Sudoku Solver AI using python and Open CV to read a Sudoku puzzle from an image and solving it. There a lot of methods to achieve this goal. Thus in this series, I have compiled the best methods I could find/research along with some hacks/tricks I learned along the way.

## Tutorial 
Access the written tutorial of [SudkoAI](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629). This article is a part of the series Sudoku Solver AI with OpenCV. \
**Part 1:** [Image Processing](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629) \
**Part 2:** [Sudoku and Cell Extraction](https://medium.com/@aditijain0424/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066) \
**Part 3:** [Solving the Sudoku](https://medium.com/@aditijain0424/part-3-solving-the-sudoku-ai-solver-13f64a090922)


## Run
```
python3 main.py
```

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
### Sudoku
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/sudoku_1.jpg) 
### Pre Processing the Image
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/grey_scale.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/processed.png)
### Sudoku Extraction
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/pre_processed.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/cropped.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/processed_sudoku.png) \


### Interpreting the Digits
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/extracted_cell.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/cell_contour.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/model.png) 
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/number.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/predicted_num.png) 
### Solving
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/sudokuboard.png) \
![](https://github.com/Joy2469/Sudoku_AI/blob/master/images/Solved.png) 



# Resources:
[Deep Learning Introduction](https://medium.com/r/?url=https%3A%2F%2Fwww.forbes.com%2Fsites%2Fbernardmarr%2F2018%2F10%2F01%2Fwhat-is-deep-learning-ai-a-simple-guide-with-8-practical-examples%2F%235a233f778d4b)<br/>
[Install Tensorflow](https://medium.com/@cran2367/install-and-setup-tensorflow-2-0-2c4914b9a265)<br/>
[Why Data Normalizing](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029)<br/>
[One-Hot Code](https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fwhy-one-hot-encode-data-in-machine-learning%2F)<br/>
[Understanding of Convolutional Neural Network (CNN)](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148%20https://www.youtube.com/watch?v=YRhxdVk_sIs)<br/>
[CNN layers](https://medium.com/r/?url=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fkeras%2Flayers%2FLayer)<br/>
[K-cross Validation](https://medium.com/r/?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DTIgfjmp-4BA)<br/>
[Plotting Graphs](https://medium.com/r/?url=https%3A%2F%2Fmatplotlib.org%2Fapi%2Fpyplot_api.html)<br/>
