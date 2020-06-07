# MSiD-Project
Classification algorithm for images from Fashion-MNIST dataset.
### Introduction
Fashion-MNIST is a data set of 28x28 greyscale images of clothing pieces from 10 categories. It contains 60,000 training examples and 10,000 test examples.I am sharing my result on this data set using a KNN algorithm as well as a CNN.
## KNN
### Methods
I calculated the distance between examlpes as euclidean distance. https://en.wikipedia.org/wiki/Euclidean_distance
### Results
I found k=7 to give the best results, by comparing few diffrent values end evaluating the results on a validation set created from 20% of test examples.
I also normalized the data by dividing the value for each pixel by 255.
I got a classification error of 0.1483, which means an accuracy of 0.8517.
