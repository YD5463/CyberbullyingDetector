import tensorflow.compat.v1 as tf
import numpy as np
from typing import List

tf.disable_v2_behavior()
eps = 1e-12


class LogisticRegression:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, num_iter=5000, learning_rate=0.001, batch_size=100,
                 print_step=1000):
        """
        weighted logistic regression using cross entropy loss function
        :param num_iter:
        :param batch_size: -1 means all
        """
        self.sess = tf.Session()
        features = X_train.shape[1]
        self.x = tf.placeholder(tf.float32, [None, features])
        y_train_variable = tf.placeholder(tf.float32, [None, 1])
        W = tf.Variable(tf.zeros([features, 1]))
        b = tf.Variable(tf.zeros([1]))
        self.y = 1 / (1.0 + tf.exp(-(tf.matmul(self.x, W) + b)))
        loss = tf.reduce_mean(-(0.2 * y_train_variable * tf.log(self.y + eps) + 0.8 * (1 - y_train_variable) * tf.log(
            1 - self.y + eps)))  # cross entropy
        update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # TODO: check other optimizers
        self.sess.run(tf.global_variables_initializer())
        np.random.shuffle(X_train)
        rows_num = X_train.shape[0]
        for i in range(0, num_iter):
            counter_step = i % (rows_num // batch_size)
            X_batch = X_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = y_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = Y_batch.reshape((Y_batch.size, 1))
            self.sess.run(update, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            loss_value = self.sess.run(loss, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            if i % print_step == 0:
                print(f"iteration {i}: loss value is: {loss_value}")

    def predict(self, X_test, thr=0.5):
        predictions = self.sess.run(self.y, feed_dict={self.x: X_test})
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0
        return predictions


class MLP:
    """
    multi level perceptron implementation using tensorflow version 1
    """

    def __init__(self, X_train: np.ndarray, y_train, layers_sizes: List[int], learning_rate=0.001, num_iter=5000,
                 batch_size=100,
                 print_step=100):
        """
        Feed Foreword Neural network using Batch gradient decent optimizer
        :param layers_sizes: len of this list need to be greater than 1
        :param num_iter:
        :param print_step: print loss value every print_step echos
        """
        self.sess = tf.Session()
        rows_num, features = X_train.shape[0], X_train.shape[1]
        self.x = tf.placeholder(tf.float32, [None, features])
        y_train_variable = tf.placeholder(tf.float32, [None, 1])
        layers_sizes = [features] + layers_sizes.copy() + [1]
        W, b = [], []
        for i, layer_size in enumerate(layers_sizes[1:]):
            W.append(tf.Variable(tf.zeros([layers_sizes[i], layer_size])))
            b.append(tf.Variable(tf.zeros(layer_size)))
        # ff
        prev_output = tf.nn.relu(tf.matmul(self.x, W[0]) + b[0])
        for layer_w, layer_b in zip(W[1:], b[1:]):
            # tf.nn.relu
            prev_output = tf.nn.relu(1 / (1.0 + tf.exp(-(tf.add(tf.matmul(prev_output, layer_w), layer_b)))))
        self.y = tf.nn.sigmoid(prev_output)
        loss = tf.reduce_mean(-(y_train_variable * tf.log(self.y + eps) + (1 - y_train_variable) * tf.log(
            1 - self.y + eps)))  # cross entropy
        update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)  # TODO: check other optimizers
        self.sess.run(tf.global_variables_initializer())
        np.random.shuffle(X_train)
        for i in range(0, num_iter):
            counter_step = i % (rows_num // batch_size)
            X_batch = X_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = y_train[counter_step * batch_size:min((counter_step + 1) * batch_size, rows_num)]
            Y_batch = Y_batch.reshape((Y_batch.size, 1))
            self.sess.run(update, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            loss_value = self.sess.run(loss, feed_dict={self.x: X_batch, y_train_variable: Y_batch})
            if i % print_step == 0:
                print(f"iteration {i}: loss value is: {loss_value}")

    def predict(self, X_test, thr=0.5):
        predictions = self.sess.run(self.y, feed_dict={self.x: X_test})
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0
        return predictions
