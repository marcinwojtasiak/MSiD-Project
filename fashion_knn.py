from tensorflow.keras.datasets.fashion_mnist import load_data
from knn import distance, sort_train_labels_knn, classification_error, p_y_x_knn

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train / 255.
x_test = x_test / 255.

k = 7

dist = distance(x_test, x_train)
y_sorted = sort_train_labels_knn(dist, y_train)
p_y_x = p_y_x_knn(y_sorted, k)
error = classification_error(p_y_x, y_test)

print("Error: ", error)
