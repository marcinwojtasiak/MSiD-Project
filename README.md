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
### Usage
To use the algorithm
## Convolutional neural network
### Methods
I used a CNN of following architecture:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
batch_normalization_1 (Batch (None, 28, 28, 1)         4         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        640       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 64)        102464    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        102464    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               803072    
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
=================================================================
Total params: 1,049,166
Trainable params: 1,048,652
Non-trainable params: 514
_________________________________________________________________
```
It has two pairs of convolutional layers, each followed by a max pooling layer and two fully connected layers. I tested few diffrent architectures and this one proved to give the best results.

Preprocessing:
Data normalization - divided value for each pixel by 255.
Data augmentation - rotation and horizontal flips

Training set - validation set: 80% - 20%

I used dropout to prevent overfitting and batch normalization to improve speed and stability of the network.

Hyperparameters aren't fine-tuned. Results could be slightly better after fine-tuning.
### Results
I achieved a loss of 0.2622 and an accuracy of 94.59%
### Usage
