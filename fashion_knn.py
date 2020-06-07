import mnist_reader
from knn import distance, sort_train_labels_knn, classification_error, p_y_x_knn

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train / 255.
X_test = X_test / 255.

k = 7

dist = distance(X_test, X_train)
y_sorted = sort_train_labels_knn(dist, y_train)
p_y_x = p_y_x_knn(y_sorted, k)
error = classification_error(p_y_x, y_test)

print("Error: ", error)