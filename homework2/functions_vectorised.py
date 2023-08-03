import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    if len(np.diag(X)[np.diag(X) >= 0]):
        return sum(np.diag(X)[np.diag(X) >= 0])
    return -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    prod = np.multiply(x[1:], x[:-1])
    if len(prod[prod % 3 == 0]) != 0:
        return np.max(prod[prod % 3 == 0])
    return -1



def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.dot(image, weights)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    X = np.repeat(x[:, 0], x[:, 1])
    Y = np.repeat(y[:, 0], y[:, 1])
    if len(X) == len(Y):
        return np.dot(X, Y)
    return -1


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    scalar_multiple=np.dot(X, np.transpose(Y))
    norm = np.dot(np.linalg.norm(X, axis=1, keepdims=True), np.transpose(np.linalg.norm(Y, axis=1, keepdims=True)))
    return np.where(norm != 0, scalar_multiple/norm, 1)
