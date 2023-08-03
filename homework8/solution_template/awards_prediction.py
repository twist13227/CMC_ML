from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from numpy import ndarray

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    # remove categorical variables

    categorical = ["genres", "directors", "filming_locations"]
    genders = ["actor_0_gender", "actor_1_gender", "actor_2_gender"]
    y_train = df_train["awards"]
    df_train = df_train.drop(["awards"], axis=1)
    train_size = df_train.shape[0]
    dataset = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    dataset = dataset.drop("keywords", axis=1)
    for category in categorical:
        dataset[category] = dataset[category].apply(
            lambda x: ["unknown"]
            if (isinstance(x, str) and x.lower() == "unknown")
            else x
        )
        dataset[category] = dataset[category].str.join(",")
    vectorizer = CountVectorizer(token_pattern=r"[a-z ]+")
    genres_vectorized = vectorizer.fit_transform(dataset["genres"]).toarray()
    directors_vectorized = vectorizer.fit_transform(dataset["directors"]).toarray()
    filming_locations_vectorized = vectorizer.fit_transform(
        dataset["filming_locations"]
    ).toarray()
    dataset.loc[:, genders] = dataset.loc[:, genders].astype("category")
    dataset = dataset.drop(categorical, axis=1)
    vectorized = pd.concat(
        [
            pd.DataFrame(genres_vectorized),
            pd.DataFrame(directors_vectorized),
            pd.DataFrame(filming_locations_vectorized),
        ],
        axis=1,
        ignore_index=True,
    )
    dataset = pd.concat([dataset, vectorized], axis=1)
    x_train = dataset.iloc[:train_size]
    x_test = dataset.iloc[train_size:]
    cat = CatBoostRegressor(
        train_dir="/tmp/catboost_info",
        learning_rate=0.024535318842786617,
        max_depth=5,
        n_estimators=1797,
        verbose=False,
        cat_features=genders,
    )
    cat.fit(x_train, y_train)
    return cat.predict(x_test)
