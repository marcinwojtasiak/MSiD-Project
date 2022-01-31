import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

M = 10


def distance(X, X_train):
    return euclidean_distances(X, X_train)


def sort_train_labels_knn(Dist, y):
    indexes = np.argsort(Dist, kind="mergesort")
    return y[indexes]


def p_y_x_knn(y, k):
    N1 = np.shape(y)[0]
    result = []
    for n1 in range(N1):
        k_nearest = []
        for i in range(k):
            k_nearest.append(y[n1][i])
        counts = np.bincount(k_nearest, minlength=M)
        result.append(counts / k)

    return result


def classification_error(p_y_x, y_true):
    N = np.size(y_true)
    m = len(p_y_x[0])
    wrong = 0
    for n in range(N):
        best = m - np.argmax(p_y_x[n][::-1]) - 1
        if best != y_true[n]:
            wrong += 1

    return wrong/N


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    best_k = None
    errors = []
    best_error = 1

    distances = distance(X_val, X_train)
    sorted_labels = sort_train_labels_knn(distances, y_train)

    for k in k_values:
        p_y_x = p_y_x_knn(sorted_labels, k)
        error = classification_error(p_y_x, y_val)
        errors.append(error)
        if error < best_error:
            best_k = k
            best_error = error

    return best_error, best_k, errors


def estimate_a_priori_nb(y_train):
    return np.bincount(y_train, minlength=4) / np.size(y_train)


def estimate_p_x_y_nb(X_train, y_train, a, b):
    N = np.shape(X_train)[0]
    D = np.shape(X_train)[1]

    y_count = np.bincount(y_train, minlength=M)
    param = np.zeros([M, D])

    for n in range(N):
        y = y_train[n]
        param[y] += X_train[n]

    param += (a - 1)
    y_count += (a + b - 2)

    for m in range(M):
        param[m] /= y_count[m]

    return param


def p_y_x_nb(p_y, p_x_1_y, X):
    p_x_1_y_rev = 1 - p_x_1_y
    X_rev = 1 - X
    result = []
    for n in range(X.shape[0]):
        success = p_x_1_y ** X[n, ]
        failure = p_x_1_y_rev ** X_rev[n, ]
        numerator = np.prod(success * failure, axis=1) * p_y
        sum = np.sum(numerator)
        result.append(numerator / sum)

    return result


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    a_length = len(a_values)
    b_length = len(b_values)

    best_a = None
    best_b = None
    best_error = 1
    errors = np.empty((a_length, b_length))

    p_y = estimate_a_priori_nb(y_train)

    for i in range(a_length):
        for j in range(b_length):
            p_x_y = estimate_p_x_y_nb(X_train, y_train, a_values[i], b_values[j])
            p_y_x = p_y_x_nb(p_y, p_x_y, X_val)
            error = classification_error(p_y_x, y_val)
            errors[i][j] = error
            if error < best_error:
                best_a = a_values[i]
                best_b = b_values[j]
                best_error = error

    return best_error, best_a, best_b, errors
