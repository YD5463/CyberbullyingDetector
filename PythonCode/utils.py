import pandas as pd
import numpy as np
import re
from collections import Counter


def get_words_count(text: str) -> (pd.Series, int):
    text = re.findall(r"(?u)\b\w\w+\b", text)
    words_count = pd.Series(Counter(text))
    words_count = words_count.drop(labels=set(words_count.keys()).difference(set(stopwords.words('english'))))
    return words_count, len(text)


class CountVectorizer:
    def __init__(self, X: pd.Series, min_df=0.0001):
        self.words_count, text_len = get_words_count(" ".join(X.ravel()))
        self.words_count = self.words_count[self.words_count > min_df * text_len]

    def transform(self, X: pd.Series):
        def helper(row_text):
            res = get_words_count(row_text)[0]
            return res.drop(set(res.keys()).difference(self.words_count.keys()))

        return X.swifter.apply(helper).fillna(0)


def train_test_split(X, y, test_size, random_state):
    # x_train, y_train, x_test, y_test
    np.random.seed(random_state)
    num_rows = X.shape[0]
    mask_train = np.zeros(num_rows, dtype=bool)
    mask_train[np.random.choice(num_rows, int(num_rows * test_size), replace=False)] = True
    return X[~mask_train, :], y[~mask_train], X[mask_train, :], y[mask_train]


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    TP = y_pred[y_pred[y_true == 0] == 0].size
    TN = y_pred[y_pred[y_true == 1] == 1].size
    FP = y_pred[y_pred[y_true == 0] == 1].size
    FN = y_pred[y_pred[y_true == 1] == 0].size
    precision_0 = TP / (TP + FP)
    precision_1 = TN / (TN + FN)
    recall_0 = TP / (TP + FN)
    recall_1 = TN / (TN + FP)
    f_score_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    f_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    print("\tprecision\trecall\tf1-score")
    print(f"0\t  {precision_0}\t {recall_0}\t {f_score_0}\n1\t  {precision_1}\t {recall_1}\t {f_score_1}")
    print(f"accuracy\t\t\t{(TP + TN) / (TP + FP + TN + FN)}")
    return np.array([[TP, FN], [FP, TN]])
