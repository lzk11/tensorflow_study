import tensorflow as tf
import numpy as np

dropout = tf.placeholder(tf.float32)
x = tf.Variable(tf.random_normal([2, 3], seed=1, stddev=1))
y = tf.nn.dropout(x, dropout)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
    print(sess.run(y, feed_dict={dropout: 0.5}))