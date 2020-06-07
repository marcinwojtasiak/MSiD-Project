import mnist_reader
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import to_categorical

model = load_model('trained_model/cnn')
model.summary()

X_test, y_test = mnist_reader.load_mnist('data\\fashion', kind='t10k')

X_test = X_test / 255.
X_test = X_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test)

score = model.evaluate(X_test, y_test)

print(score)
file = open('trained_model/history', 'rb')
history = pickle.load(file)
file.close()


def plot_train_val(history):
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
    plt.show()


plot_train_val(history)
