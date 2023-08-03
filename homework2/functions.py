from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X.
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    sum = None
    for i in range(min(len(X[0]), len(X))):
        if X[i][i] >= 0:
            if sum == None:
                sum = 0
            sum += X[i][i]
    if sum is not None:
        return sum
    return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return sorted(x) == sorted(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max_prod = None
    for i in range(len(x) - 1):
        prod = x[i] * x[i + 1]
        if prod % 3 == 0:
            if max_prod is None:
                max_prod = prod
            if prod > max_prod:
                max_prod = prod
    if max_prod is not None:
        return max_prod
    return -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    #(height, width, channels_num)
    parameters = (len(image), len(image[0]), len(weights))
    result = [[0] * parameters[1] for i in range(parameters[0])]
    for k in range(parameters[2]):
        for i in range(parameters[0]):
            for j in range(parameters[1]):
                result[i][j] += image[i][j][k] * weights[k]
    return result


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    normal_x = []
    for i in range(len(x)):
        for j in range(x[i][1]):
            normal_x.append(x[i][0])
    normal_y = []
    for i in range(len(y)):
        for j in range(y[i][1]):
            normal_y.append(y[i][0])

    if len(normal_x) == len(normal_y):
        return sum(normal_x[i] * normal_y[i] for i in range(len(normal_x)))
    return -1


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = [[1 for j in range(len(Y))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            scalar_multiple = 0
            sum_xx = 0
            sum_yy = 0
            for c in range(len(X[i])):
                scalar_multiple += X[i][c] * Y[j][c]
                sum_xx += X[i][c] ** 2
                sum_yy += Y[j][c] ** 2
            if sum_xx != 0 and sum_yy != 0:
                res[i][j] = scalar_multiple / ((sum_xx * sum_yy) ** 0.5)
    return res
