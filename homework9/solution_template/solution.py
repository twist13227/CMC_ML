import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        """
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        """
        super().__init__()
        self.n_clusters = n_clusters

        # Ваш код здесь:＼(º □ º l|l)/

    def fit(self, data, labels):
        """
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        """
        # Ваш код здесь:＼(º □ º l|l)/

        return self

    def predict(self, data):
        """
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        """
        # Ваш код здесь:＼(º □ º l|l)/

        return predictions

    def _best_fit_classification(self, cluster_labels, true_labels):
        """
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        """
        # Ваш код здесь:＼(º □ º l|l)/
        unique_cluster_labels = np.unique(cluster_labels)
        unique_true_labels, count_of_true_labels = np.unique(
            true_labels, return_counts=True
        )
        most_frequent_label = (
            unique_true_labels[1:][np.argmax(count_of_true_labels[1:])]
            if unique_true_labels[0] == -1
            else unique_true_labels[np.argmax(count_of_true_labels)]
        )
        mapping = np.full(self.n_clusters, most_frequent_label)
        for clucter_label in unique_cluster_labels:
            objects = true_labels[cluster_labels == clucter_label]
            labeled = np.delete(objects, np.where(objects == -1))
            if labeled.size > 0:
                cur_labels, cur_counts = np.unique(labeled, return_counts=True)
                mapping[clucter_label] = cur_labels[np.argmax(cur_counts)]
        return mapping, mapping[cluster_labels]
