#!/usr/bin/env python
# -*- cooding: utf-8 -*-
# 预测产量问题：构造不同的损失函数
#

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_-input')

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = conv2d(x, w1)
y = tf.nn.relu(conv2d(a, w2))

loss_more = 1
loss_less = 10

loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                               (y - y_) * loss_less,
                               (y_ - y) * loss_more))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)

dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]
print(X)
print(Y)

tf.train.exponential_decay()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000

    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start: end],
                            y_: Y[start: end]})
        if i % 1000 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("第{i}次迭代的损失:{total_loss}".format(i=i, total_loss=total_loss))

    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(conv2d(w1, w2)))

