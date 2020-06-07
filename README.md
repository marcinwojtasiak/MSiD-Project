# MSiD-Project
Classification algorithm for images from Fashion-MNIST dataset.
### Introduction
Fashion-MNIST is a data set of 28x28 greyscale images of clothing pieces from 10 categories. It contains 60,000 training examples and 10,000 test examples.I am sharing my result on this data set using a KNN algorithm as well as a CNN.
## KNN
### Methods
I calculated the distance between examples as euclidean distance.
I also normalized the data by dividing the value for each pixel by 255.  
### Results
I found k=7 to give the best results, by comparing few diffrent values end evaluating the results on a validation set created from 20% of test examples.  
I got a classification error of 0.1483, which means an accuracy of 0.8517.  
It's a result comparable to those on zalando research benchmark which have an average accuracy of 0.8543.
### Usage
To use the algorithm you will need these files: [data/fashion](https://github.com/marcinwojtasiak/MSiD-Project/tree/master/data/fashion), [knn.py](https://github.com/marcinwojtasiak/MSiD-Project/blob/master/knn.py), [fashion_knn.py](https://github.com/marcinwojtasiak/MSiD-Project/blob/master/fashion_knn.py) and [mnist_reader.py](https://github.com/marcinwojtasiak/MSiD-Project/blob/master/mnist_reader.py). Then just run the fashion_knn.py script.
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
It has two pairs of convolutional layers, each followed by a max pooling layer and then two fully connected layers. I tested few diffrent architectures and this one proved to give the best results.

Preprocessing:  
Data normalization - divided value for each pixel by 255.  
Data augmentation - rotation and horizontal flips.  

Training set - validation set: 80% - 20%

I used dropout to prevent overfitting and batch normalization to improve speed and stability of the network.

Hyperparameters aren't fine-tuned. Results could be slightly better after fine-tuning.
### Results
I achieved a loss of 0.2578 and an accuracy of 94.31%  
<img src="./images/plots.png">
### Usage
To create and train the CNN, you need [data/fashion](https://github.com/marcinwojtasiak/MSiD-Project/tree/master/data/fashion), [mnist_reader](https://github.com/marcinwojtasiak/MSiD-Project/blob/master/mnist_reader.py) and [cnn.py](https://github.com/marcinwojtasiak/MSiD-Project/blob/master/cnn.py) files. The cnn.py script will train the model and save it in folder trained_model as "cnn" using kers.model.save_model, together with history dictionary from history object as "history" using pickle.
The scrpit uses numpy, sklearn, matplotlib and tensorflow.
There is my trained CNN in folder [trained_model](https://github.com/marcinwojtasiak/MSiD-Project/tree/master/trained_model) which can be loaded using keras.model.load_model, there is also it's training history which you can load with pickle and use to plot loss and accuracy.  
test_model.py will load those and print model summary, evaluate the model and plot the history. It uses pickle, matplotlib and tensorflow.
