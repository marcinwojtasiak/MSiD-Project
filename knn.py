# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

M = 10


def distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    return euclidean_distances(X, X_train)


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    indexes = np.argsort(Dist, kind="mergesort")
    return y[indexes]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
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
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    N = np.size(y_true)
    m = len(p_y_x[0])
    wrong = 0
    for n in range(N):
        best = m - np.argmax(p_y_x[n][::-1]) - 1
        if best != y_true[n]:
            wrong += 1

    return wrong/N


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
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
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    return np.bincount(y_train, minlength=4) / np.size(y_train)


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
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
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
#    X = X.toarray()
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
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
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
