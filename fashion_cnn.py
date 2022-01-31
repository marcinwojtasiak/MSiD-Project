from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Accuracy
    ax1.set_title('Model accuracy')
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')

    # Loss
    ax2.set_title('Model loss')
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.set_size_inches(20, 5)
    fig.savefig('images/plots.png')


# get data
(x_train, y_train), (x_test, y_test) = load_data()

# reshape data
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# split data into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

# create model
model = Sequential()

model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28), padding='same'))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# train and save model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=256)
model.save('trained_model')

# evaluate model
score = model.evaluate(x_test, y_test)
print(score)

# plot loss and acc
plot_history(history.history)
