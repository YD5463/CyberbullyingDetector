import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LogisticRegression:
    def __init__(self,X_train,y_train,num_iter=5000,learning_rate=0.001):
        self.sess = tf.Session()
        features = X_train.shape[1]
        eps = 1e-12
        self.x = tf.placeholder(tf.float32, [None, features])
        y_ = tf.placeholder(tf.float32, [None, 1])
        W = tf.Variable(tf.zeros([features, 1]))
        b = tf.Variable(tf.zeros([1]))
        self.y = 1 / (1.0 + tf.exp(-(tf.matmul(self.x, W) + b)))
        loss = tf.reduce_mean(-(y_ * tf.log(self.y + eps) + (1 - y_) * tf.log(1 - y + eps)))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self.sess.run(tf.global_variables_initializer())
        for i in range(0, num_iter):
            _, loss_value = self.sess.run([optimizer, loss], feed_dict={x: X_train.to_numpy(), y_: y_train})
            if i % 1000 == 0:
                print(f"iteration {i}: loss value is: {loss_value}")

    def predict(self,X_test):
        predictions = self.sess.run(self.y, feed_dict={self.x: X_test.to_numpy()})
        thr = 0.05
        predictions[predictions >= thr] = 1
        predictions[predictions < thr] = 0

