import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    unique_labels, labels_counts = np.unique(labels, return_counts=True)
    if (unique_labels.shape[0] == 1):
        return 0
    dists = sklearn.metrics.pairwise_distances(x)
    matrix = np.zeros((x.shape[0], len(unique_labels)))
    mask = np.zeros((x.shape[0], len(unique_labels)), dtype=bool)
    counts = np.zeros(x.shape[0])
    for i, label in enumerate(unique_labels):
        matrix[:, i] = np.sum(dists[:, labels == label], axis=1)
        mask[:, i] = labels == label
        counts[labels == label] = labels_counts[i]

    s = matrix[mask]
    s = s[counts != 1] / (counts[counts != 1] - 1)
    d = matrix / labels_counts
    d = d[mask == 0]
    d = np.min(np.reshape(d, (x.shape[0], len(unique_labels) - 1)), axis=1)
    d = d[counts != 1]
    ds_max = np.max(np.array((d, s)), axis=0)
    sil = np.zeros(x.shape[0])
    sil[counts != 1] = (d - s) / np.where(ds_max == 0, 1, ds_max)
    return np.mean(sil)


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    # Ваш код здесь:＼(º □ º l|l)/
    repeated_true_labels = np.repeat(true_labels[..., np.newaxis], true_labels.shape[0], axis=1)
    repeated_predicted_labels = np.repeat(predicted_labels[..., np.newaxis], predicted_labels.shape[0], axis=1)
    equal_of_true_labels = np.equal(repeated_true_labels, true_labels)
    equal_of_predicted_labels = np.equal(repeated_predicted_labels, predicted_labels)
    sum_true = np.sum(equal_of_true_labels, axis=0)
    sum_predicted = np.sum(equal_of_predicted_labels, axis=0)
    matrix_mult = np.multiply(equal_of_true_labels, equal_of_predicted_labels)
    precision = np.mean(np.divide(np.sum(matrix_mult, axis=0), sum_predicted))
    recall = np.mean(np.divide(np.sum(matrix_mult, axis=0), sum_true))
    return 2 * (precision * recall) / (precision + recall)
