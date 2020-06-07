import mnist_reader
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# get data
X_train, y_train = mnist_reader.load_mnist('data\\fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data\\fashion', kind='t10k')

# reshape to fit cnn
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# data augmentation
datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, fill_mode='nearest')


def image_augmentation(image, num_of_augments):
    new = []
    image = image.reshape(1, 28, 28, 1)
    for img in datagen.flow(image, batch_size=1):
        new.append(img)
        num_of_augments -= 1
        if num_of_augments == 0:
            break
    return new


def preprocess_data(images, labels, augment_data=False, num_of_augments=1):
    result_X = []
    result_y = []
    for x, y in zip(images, labels):
        x = x / 255.0

        if augment_data:
            aug_img = image_augmentation(x, num_of_augments)
            for i in aug_img:
                result_X.append(i.reshape(28, 28, 1))
                result_y.append(y)

        result_X.append(x)
        result_y.append(y)
    print("Samples after preprocessing: ", len(result_X))
    return np.array(result_X), np.array(result_y)


# pre process data
X_train, y_train = preprocess_data(X_train, y_train, True, 2)
X_test, y_test = preprocess_data(X_test, y_test)


# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1337)

# create model
model = Sequential()

# add layers
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# train and save model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=256)
model.save('trained_model/cnn')
# save history
from pickle import dump
file = open('trained_model/history', 'wb')
dump(history.history, file)
file.close()

# evaluate model
score = model.evaluate(X_test, y_test, batch_size=256)
print(score)
