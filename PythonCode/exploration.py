import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def most_common_words(df: pd.DataFrame):
    """" most common words per class"""
    pass


def label_balance(df: pd.DataFrame):
    fig1, ax1 = plt.subplots()
    ax1.pie(df["oh_label"].value_counts(), explode=(0, 0.1), labels=["bullying", "ok"], autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()


def features_relationship(df: pd.DataFrame):
    sns.heatmap(df.corr())


def explore():
    df = pd.read_csv("Data/cleaned.csv",index_col=0)
    print(df.head())
    print(df.columns)
    print(df.shape)
# TODO: add more


