import nltk
import os
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from enchant.checker import SpellChecker

import re


def merge_datasets() -> pd.DataFrame:
    df = pd.DataFrame()
    data_path = './Data/Source1'
    for filename in os.listdir(data_path):
        df1 = pd.read_csv(f'{data_path}/{filename}')
        df1 = df1[['Text', 'oh_label']]
        df = pd.concat([df, df1], axis=0)
    return df


def add_punctuation_and_stopwords_features(df: pd.DataFrame) -> pd.DataFrame:
    for ch in list(string.punctuation) + stopwords.words('english'):
        df[ch] = df['Text'].astype(str).map(lambda s: s.count(ch) / len(s))
    return df


def add_count_misspell_feature(df: pd.DataFrame) -> pd.DataFrame:
    def helper(data: str) -> float:
        spell = SpellChecker("en_US", data)
        counter = 0
        for _ in spell:
            counter += 1
        return counter / len(data)
    df["misspell_count"] = df["Text"].swifter.apply(helper)


def process_row(row):
    lemmatizer = WordNetLemmatizer()
    # row = row.translate(str.maketrans('', '', string.punctuation))
    row = " ".join(lemmatizer.lemmatize(w) for w in nltk.wordpunct_tokenize(row))
    return " ".join(w for w in nltk.wordpunct_tokenize(row))  # word exist in corpus and remove stop words


def to_one_hot_rep(df: pd.DataFrame) -> pd.DataFrame:
    cv = CountVectorizer()
    data_cv = cv.fit_transform(df.Text)
    data_cv = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_cv["oh_label"] = df["oh_label"]
    return data_cv


def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    res = df.sum(axis=0)
    res = res[res > res.median()]
    ls = res.index.to_list()
    del ls[0]
    return df[df.columns.intersection(ls)]


def preprocess(train_part=0.7, use_cache=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    cleaned_output_path = "./Data/Source1/cleaned.csv"
    if use_cache and os.path.isfile(cleaned_output_path):
        df = pd.read_csv(cleaned_output_path)
    else:
        df = merge_datasets()
        df = add_punctuation_and_stopwords_features(df)
        df = add_count_misspell_feature(df)
        # df["Text"] = df["Text"].apply(process_row)
        # df = to_one_hot_rep(df)
        df.to_csv(cleaned_output_path)
    label_name = "oh_label"
    x = df.drop(label_name, axis=1)
    y = df[label_name]
    num_rows = x.shape[0]
    mask_train = np.zeros(num_rows, dtype=bool)
    mask_train[np.random.choice(num_rows, int(num_rows * train_part), replace=False)] = True
    return x[mask_train, :], y[mask_train], x[~mask_train, :], y[~mask_train]
