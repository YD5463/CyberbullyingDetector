import tensorflow.compat.v1 as tf
import numpy as np
from typing import List

tf.disable_v2_behavior()
eps = 1e-2


def save_loss(data: list, filename: str):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


class LogisticRegression:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, epoch=30, learning_rate=0.001, batch_size=100):
        """
        weighted logistic regression using cross entropy loss function
        :param num_iter:
        :param batch_size: -1 means all
        """
        self.losses = []
        self.sess = tf.Session()
        features = X_train.shape[1]
        self.x = tf.placeholder(tf.float64, [None, features])
        y_train_variable = tf.placeholder(tf.float64, [None, 1])
        W = tf.Variable(tf.random.ones([features, 1],dtype=tf.float64))
        b = tf.Variable(tf.random.ones([1],dtype=tf.float64))
        self.y = 1 / (1.0 + tf.exp(-(tf.matmul(self.x, W) + b)))
        w1_weight = (y_train == 1).sum()    # for imbalance data
        w0_weight = (y_train == 0).sum()
        loss = tf.reduce_mean(-(((w1_weight+w0_weight)/w1_weight) * y_train_variable * tf.log(self.y + eps) + ((w1_weight+w0_weight)/w0_weight) * (1 - y_train_variable) * tf.log(
            1 - self.y + eps)))  # cross entropy
        update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # TODO: check other optimizers
        self.sess.run(tf.global_variables_initializer())
        np.random.shuffle(X_train)
        rows_num = X_train.shape[0]
        for i in range(0, epoch * (rows_num//batch_size)):
            counter_step = i % (rows_num // batch_size)
            X_batch = X_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = y_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = Y_batch.reshape((Y_batch.size, 1))
            self.sess.run(update, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            loss_value = self.sess.run(loss, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            if i % (rows_num//batch_size) == 0:
                print(f"iteration {i}: loss value is: {loss_value}")
                self.losses.append(loss_value)
        save_loss(self.losses, filename="LogisticRegression.txt")

    def predict(self, X_test, thr=0.5):
        predictions = self.sess.run(self.y, feed_dict={self.x: X_test})
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0
        return predictions


class MLP:
    """
    multi level perceptron implementation using tensorflow version 1
    """

    def __init__(self, x_train: np.ndarray, y_train, layers_sizes=(200, 100), learning_rate=0.1, epoch=50,
                 batch_size=100):
        """
        Feed Foreword Neural network using Batch gradient decent optimizer
        :param layers_sizes: len of this list need to be greater than 1
        :param num_iter:
        :param print_step: print loss value every print_step echos
        """
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=1)
        self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
        self.losses = []
        self.test_loss = []
        lr = learning_rate
        learning_rate_tensor = tf.placeholder(tf.float64, shape=[])  # tensor for implement dynamic learning rate
        rows_num, features = x_train.shape[0], x_train.shape[1]
        self.x = tf.placeholder(tf.float64, [None, features])
        y_train_variable = tf.placeholder(tf.float64, [None, 1])
        layers_sizes = [features] + layers_sizes.copy() + [1]
        W, b = [], []
        for i, layer_size in enumerate(layers_sizes[1:]):
            W.append(tf.Variable(tf.random.uniform([layers_sizes[i], layer_size], dtype=tf.float64)))
            b.append(tf.Variable(tf.random.uniform([layer_size], dtype=tf.float64)))
        # ff
        prev_output = tf.nn.relu(tf.matmul(self.x, W[0]) + b[0])
        for layer_w, layer_b in zip(W[1:-1], b[1:-1]):
            print(prev_output)
            prev_output = tf.nn.relu(tf.add(tf.matmul(prev_output, layer_w), layer_b))
        print(prev_output)
        self.y = tf.nn.sigmoid(tf.add(tf.matmul(prev_output, W[-1]), b[-1]))
        print(self.y)
        w1_weight = (y_train == 1).sum()  # for imbalance data
        w0_weight = (y_train == 0).sum()
        # pos_weight multiple the 1 label => targets * -log(sigmoid(logits)) * pos_weight
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(labels=y_train_variable, logits=tf.cast(self.y, tf.float64),
                                                     pos_weight=tf.constant((w1_weight + w0_weight) / w1_weight,
                                                                            tf.float64)))
        update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_tensor).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        rows_num = x_train.shape[0]
        for i in range(0, epoch * (rows_num // batch_size)):
            counter_step = i % (rows_num // batch_size)
            X_batch = x_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = y_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = Y_batch.reshape((Y_batch.size, 1))
            self.sess.run(update, feed_dict={self.x: X_batch, y_train_variable: Y_batch, learning_rate_tensor: lr})
            self.sess.run(loss, feed_dict={self.x: X_batch, y_train_variable: Y_batch, learning_rate_tensor: lr})
            loss_value = self.sess.run(loss,
                                       feed_dict={self.x: X_batch, y_train_variable: Y_batch, learning_rate_tensor: lr})
            loss_value_test = self.sess.run(loss,
                                            feed_dict={self.x: x_test, y_train_variable: y_test.reshape(y_test.size, 1),
                                                       learning_rate_tensor: lr})
            if i % (rows_num // batch_size) == 0:
                index = i // (rows_num // batch_size)
                print(loss_value_test)
                print(f"The learning rate is: {lr}")
                print(f"iteration {index}: loss value is: {loss_value}")
                self.losses.append(loss_value)
                self.test_loss.append(loss_value_test)
                if self.losses[index] > self.losses[index - 1]:  # check if the loss divergence
                    print("=====change learning rate=========")
                    lr = lr / 10
        save_loss(self.losses, filename="MLP_train_error.txt")
        save_loss(self.test_loss, filename="MLP_test_error.txt")

    def predict(self, X_test, thr=0.5):
        predictions = self.sess.run(self.y, feed_dict={self.x: X_test})
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0
        return predictions

