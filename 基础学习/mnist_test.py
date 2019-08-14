import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8

LEARNING_RATE_DECAY  = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(conv2d(input_tensor, weight1) + biases1)
        return conv2d(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(
            conv2d(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return conv2d(layer1, avg_class.average(weight2)) + avg_class.average(biases2)


def inference2(input_tensor, reuse=False):
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable(
            name="weights",
            shape=[INPUT_NODE, LAYER_NODE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(
            name='biases',
            shape=[LAYER_NODE],
            initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(conv2d(input_tensor, weights) + biases)

    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable(
            name='weights',
            shape=[LAYER_NODE, OUTPUT_NODE],
            initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable(
            name="biases",
            shape=[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))
        layer2 = conv2d(layer1, weights) + biases
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))
    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weight1, biases1, weight2, biases2)

    # 定义存储训练轮数的全局变量， 是不可以被训练的变量
    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_average_op = variable_average.apply(tf.trainable_variables())

    average_y = inference(x, variable_average, weight1, biases1, weight2, biases2)

    cross_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entry_mean = tf.reduce_mean(cross_entry)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weight1) + regularizer(weight2)

    loss = cross_entry_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validation_feed = {x: mnist.validation.images,
                           y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validation_feed)
                print("After %d training steps, validation accuracy using average model is %g "
                      %(i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            sess.run(train_op, feed_dict={x: xs, y_:ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("test accuracy using average model is %g" % (test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()


tf.nn.dropout






