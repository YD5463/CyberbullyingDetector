import pandas as pd


def most_common_words(df: pd.DataFrame):
    """" most common words per class"""
    pass


def explore():
    df = pd.read_csv("Data/cleaned.csv",index_col=0)
    print(df.head())
    print(df.columns)
    print(df.shape)
# TODO: add more

