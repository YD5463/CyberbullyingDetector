import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import tensorflow.compat.v1 as tf
from collections import Counter
import re
import seaborn as sns

tf.disable_v2_behavior()

# --------------- Consts -------------------
eps = 0.2  # 1e-2


# ---------------- Utils -------------------------


def train_test_split(X, y, test_size, random_state):
    # x_train, y_train, x_test, y_test
    np.random.seed(random_state)
    num_rows = X.shape[0]
    mask_train = np.zeros(num_rows, dtype=bool)
    mask_train[np.random.choice(num_rows, int(num_rows * test_size), replace=False)] = True
    return X[~mask_train], X[mask_train], y[~mask_train], y[mask_train]


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


def over_sample(x_train, y_train):
    return np.concatenate((np.random.choice(x_train[y_train == 0], x_train[y_train == 1].size), x_train[y_train == 1]),
                          axis=None)


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


def preprocess():
    df = pd.read_csv("../Data/ver1.csv", index_col=False)
    df = df.dropna()
    df.drop(["Unnamed: 0", "index"], axis=1, inplace=True)
    X = df["Text"].swifter.apply(lambda text: re.sub(r'\d+', '', text).lower())
    y = df["oh_label"]
    x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state=1)
    vectorizer = CountVectorizer(pd.Series(x_train))
    x_train = vectorizer.transform(pd.Series(x_train))
    x_test = vectorizer.transform(pd.Series(x_test))
    mean, std = x_train.mean(), x_train.std()
    x_train = x_train.apply(lambda row: (row - mean) / std, axis=1)
    x_test = x_test.apply(lambda row: (row - mean) / std, axis=1)
    return x_train,x_test,y_train,y_test


# --------------------------- Models -------------------------------


def save_loss(data: list, filename: str):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def predict(sess, X_test, x_placeholder, y_final, y_test, thr=0.5):
    predictions = sess.run(tf.nn.sigmoid(sess.run(y_final, feed_dict={x_placeholder: X_test})))
    predictions[predictions >= thr] = 1
    predictions[predictions < thr] = 0
    cc = classification_report(y_test, predictions)
    sns.heatmap(cc, annot=True, fmt="g", cmap='Blues')
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.show()
    return predictions


def build_logisticRegression(features: int):
    x = tf.placeholder(tf.float64, [None, features])
    y_train_variable = tf.placeholder(tf.float64, [None, 1])
    W = tf.Variable(tf.random.uniform([features, 1], dtype=tf.float64))
    b = tf.Variable(tf.random.uniform([1], dtype=tf.float64))
    return tf.add(tf.matmul(x, W), b)


def train(sess, final_y, y_train_variable, x_placeholder, x_train, y_train, x_validation, y_validation, adaptive=True,
          learning_rate=0.1, epoch=30, batch_size=200, model_name="MLP"):
    losses = []
    valid_loss = []
    learning_rate_tensor = tf.placeholder(tf.float64, shape=[])  # tensor for implement dynamic learning rate
    lr = learning_rate
    w1_weight = (y_train == 1).sum()
    w0_weight = (y_train == 0).sum()
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y_train_variable,
                                                                   logits=tf.cast(final_y, tf.float64),
                                                                   pos_weight=tf.constant(
                                                                       (w1_weight + w0_weight) / w1_weight,
                                                                       tf.float64)))
    update = tf.train.AdamOptimizer(learning_rate_tensor).minimize(loss)
    sess.run(tf.global_variables_initializer())
    rows_num = x_train.shape[0]
    for i in range(epoch):
        for counter_step in range(rows_num // batch_size):
            X_batch = x_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = y_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            sess.run(update, feed_dict={x_placeholder: X_batch, y_train_variable: Y_batch, learning_rate_tensor: lr})
        loss_value = sess.run(loss,
                              feed_dict={x_placeholder: x_train, y_train_variable: y_train, learning_rate_tensor: lr})
        loss_value_validation = sess.run(loss, feed_dict={x_placeholder: x_validation, y_train_variable: y_validation,
                                                          learning_rate_tensor: lr})
        print(f"Iteration {i + 1}, loss = {loss_value}")
        losses.append(loss_value)
        valid_loss.append(loss_value_validation)
        if adaptive and losses[i] > losses[i - 1]:  # check if the loss divergence
            print("=====change learning rate=========")
            lr = lr / 10
    save_loss(losses, filename=f"{model_name}_train_error.txt")
    save_loss(valid_loss, filename=f"{model_name}_validation_error.txt")
    return losses, valid_loss


def build_MLP(x_train,x_test,x_validation,y_validation,y_train,y_test):
    config = tf.ConfigProto(device_count={"CPU": 8}, inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess = tf.Session(config=config)
    features = x_train.shape[1]
    x_placeholder = tf.placeholder(tf.float64, [None, features])
    y_train_variable = tf.placeholder(tf.float64, [None, 1])

    # 1000,500,100,50
    W = tf.Variable(tf.random.uniform([features, 1000], dtype=tf.float64))
    b = tf.Variable(tf.random.uniform([1000], dtype=tf.float64))
    y = tf.add(tf.matmul(x_placeholder, W), b)

    W1 = tf.Variable(tf.random.uniform([1000, 500], dtype=tf.float64))
    b1 = tf.Variable(tf.random.uniform([500], dtype=tf.float64))
    y1 = tf.add(tf.matmul(y, W1), b1)

    W2 = tf.Variable(tf.random.uniform([500, 100], dtype=tf.float64))
    b2 = tf.Variable(tf.random.uniform([100], dtype=tf.float64))
    y2 = tf.add(tf.matmul(y1, W2), b2)

    W3 = tf.Variable(tf.random.uniform([100, 50], dtype=tf.float64))
    b3 = tf.Variable(tf.random.uniform([50], dtype=tf.float64))
    y3 = tf.add(tf.matmul(y2, W3), b3)

    W4 = tf.Variable(tf.random.uniform([50, 1], dtype=tf.float64))
    b4 = tf.Variable(tf.random.uniform([1], dtype=tf.float64))
    y_final = tf.add(tf.matmul(y3, W4), b4)

    train(sess, y_final, y_train_variable, x_placeholder,x_train,y_train,x_validation, y_validation,epoch=100)
    predict(sess, x_test, x_placeholder, y_final,y_test)


# --------------------------- Plots ---------------------------

def plot_loss(losses):
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss value", fontsize=18)
    plt.title("Loss Function", fontsize=25)
    plt.plot(losses)
    plt.show()


def plot_all_losses():
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    data_train = np.loadtxt("train_error.txt")
    data_validation = np.loadtxt("validation_error.txt")
    plt.plot(data_train, color='g', label='train_error')
    plt.plot(data_validation, color='k', label='validation_error')
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss value", fontsize=18)
    plt.title("Loss Function", fontsize=25)
    plt.legend()
    plt.show()


def plot_feature_importance(X,y,label:int):
    plt.figure(figsize=(20, 15))
    data = X[y == label].sum().sort_values(ascending=False)[:50]
    sns.set_style("whitegrid")
    sns.barplot(list(data.index), data.values)
    plt.xticks(rotation=60, fontsize=15)
    plt.show()


def plot_logitic_weights(weights: np.ndarray,columns_names:np.ndarray,thr = 0.5):
    plt.figure(figsize=(45, 30))
    plt.bar(columns_names[weights > thr], weights[weights > thr])
    plt.xticks(fontsize=40)
    plt.show()


def try_different_thresholds(pred, y_test):
    for thr in [0.5, 0.6, 0.7, 0.8]:
        new_pred = pred.copy()
        new_pred[pred > thr] = 1
        new_pred[pred <= thr] = 0
        classification_report(y_test, new_pred)


