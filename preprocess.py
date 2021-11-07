import nltk
import os
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from enchant.checker import SpellChecker
# from nltk.stem import WordNetLemmatizer
# import re
import swifter


def merge_datasets(data_path='./Data/Source1') -> pd.DataFrame:
    df = pd.DataFrame()
    for filename in os.listdir(data_path):
        df1 = pd.read_csv(f'{data_path}/{filename}')
        df1 = df1[['Text', 'oh_label']]
        df = pd.concat([df, df1], axis=0)
    df["Text"] = df["Text"].astype(str)
    df = df.reset_index(drop=True)
    print("database merged successfully!")
    return df


def add_punctuation_stopwords_curse_features(df: pd.DataFrame) -> pd.DataFrame:
    def get_curses():
        lst = []
        with open("english_curse.csv") as curses_file:
            for curse in curses_file.readlines():
                lst.append(curse.replace("\n", ""))
        return lst

    curses = get_curses()
    features = list(string.punctuation) + list(stopwords.words('english')) + curses
    # new_features_cols = []
    for ch in features:
        df[ch] = df['Text'].astype(str).apply(lambda s: s.count(ch) / len(s))
    print("add_punctuation_and_stopwords_features successfully!")
    return df


def add_count_misspell_feature(df: pd.DataFrame) -> pd.DataFrame:
    def helper(data: str) -> float:
        spell = SpellChecker("en_US", data)
        counter = 0
        for _ in spell:
            counter += 1
        return counter / len(data)

    misspell_count = df["Text"].swifter.apply(helper).rename("misspell_count")
    df = pd.concat([df, misspell_count], axis=1)
    print("add_count_misspell_feature successfully!")
    return df


def add_avg_word_len_feature(df: pd.DataFrame) -> pd.DataFrame:
    avg_word_len = df["Text"].astype(str).swifter.apply(
        lambda s: pd.Series(nltk.word_tokenize(s)).map(len).mean()).rename("avg_word_len")
    df = pd.concat([df, avg_word_len], axis=1)
    print("add_avg_word_len_feature successfully!")
    return df


def add_avg_sentence_len_feature(df: pd.DataFrame) -> pd.DataFrame:
    sentence_count = df["Text"].astype(str).swifter.apply(
        lambda text: pd.Series(nltk.sent_tokenize(text)).map(lambda sent: len(nltk.word_tokenize(sent))).mean()) \
        .rename("sentence_count")

    df = pd.concat([df, sentence_count], axis=1)
    print("add_avg_sentence_len_feature successfully!")
    return df


def add_uppercase_count_feature(df: pd.DataFrame) -> pd.DataFrame:
    uppercase_count = df['Text'].str.findall(r'[A-Z]').str.len().rename("uppercase_count")/df["Text"].str.len()
    df = pd.concat([df, uppercase_count], axis=1)
    print("add_uppercase_count_feature successfully!")
    return df


def add_pos_features(df: pd.DataFrame) -> pd.DataFrame:
    def group_pos(tag):
        groups = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "verb": ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                  "adverb": ['RB', 'RBR', 'RBS'], "adjective": ['JJ', 'JJR', 'JJS']}
        for key, group in groups.items():
            if tag in group:
                return key
        return None

    features = df["Text"].swifter.apply(lambda s: pd.Series([x[1] for x in nltk.pos_tag(nltk.word_tokenize(s))]).
                                        apply(group_pos).value_counts(normalize=True).copy())
    print("add_pos_features successfully!")
    features = features.fillna(0)
    return pd.concat([df, features], axis=1)


def to_one_hot_rep(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    cv = CountVectorizer()
    data_cv = cv.fit_transform(df[col_name])
    data_cv = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_cv["oh_label"] = df["oh_label"]
    return data_cv


# def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
#     res = df.sum(axis=0)
#     res = res[res > res.median()]
#     ls = res.index.to_list()
#     del ls[0]
#     return df[df.columns.intersection(ls)]

def bug_fix(df:pd.DataFrame,label_name:str)->pd.DataFrame:
    print("loaded!")
    df["uppercase_count"] /= df["Text"].str.len()
    print("fix broken col")
    ignored_columns = ["Text", label_name]
    X_df = df.drop(ignored_columns,axis=1)
    normalized_X_df = (X_df-X_df.mean())/X_df.std()
    df = pd.concat([df[ignored_columns],normalized_X_df],axis=1)
    print("normalized")
    return df


def preprocess(train_part=0.7, use_cache=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    print("preprocess...")
    cleaned_output_path = "./Data/cleaned.csv"
    label_name = "oh_label"
    if use_cache and os.path.isfile(cleaned_output_path):
        df = pd.read_csv(cleaned_output_path,index_col=0)
        # df = bug_fix(df,label_name)
    else:
        df = merge_datasets()
        df = add_pos_features(df)
        df = add_punctuation_stopwords_curse_features(df)
        df = add_uppercase_count_feature(df)
        df = add_avg_word_len_feature(df)
        df = add_count_misspell_feature(df)
        df = add_avg_sentence_len_feature(df)
        # df["Text"] = df["Text"].apply(process_row)
        # df = to_one_hot_rep(df)
        df.to_csv(cleaned_output_path)
        print("Saved")
    x = df.drop(label_name, axis=1).values
    y = df[label_name].values
    num_rows = x.shape[0]
    mask_train = np.zeros(num_rows, dtype=bool)
    mask_train[np.random.choice(num_rows, int(num_rows * train_part), replace=False)] = True
    print(mask_train.shape, x.shape, y.shape)
    return x[mask_train, :], y[mask_train], x[~mask_train, :], y[~mask_train]
