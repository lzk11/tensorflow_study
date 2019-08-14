import tensorflow as tf
import logging
import time
import random
import numpy as np
import os
import sys
import pickle


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a')
class_num = 10
image_size = 32
image_channels = 3
iterations = 500
batch_size = 250
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_logs'
model_save_path = './model/'


def download_data():
    dirname = 'cifar10-dataset'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './CAFIR-10_data/cifar-10-python.tar.gz'
    fpath = os.path.join('.', dirname)
    download_flag = True
    if os.path.exists(fpath) or os.path.isfile(fname):
        logging.info('dataset is exists, need not to download!')
        download_flag = False
    if download_flag:
        logging.info('download dataset from uri:%s', origin)

        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        logging.info('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    logging.info("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(os.path.join(data_dir, files[0]))
    for f in files[1:]:
        data_n, labels_n = load_data_one(os.path.join(data_dir, f))
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    logging.info('raw data shape:{}'.format(np.shape(data[1])))
    data = data.reshape([-1, image_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    logging.info("======Loading data======")
    download_data()
    data_dir = './cifar10-dataset'
    meta = unpickle(os.path.join(data_dir, 'batches.meta'))
    logging.info('meta:%s', meta)
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    logging.info("Train data:{} {}".format(np.shape(train_data), np.shape(train_labels)))
    logging.info("Test data:{} {}".format(np.shape(test_data), np.shape(test_labels)))
    logging.info("======Load finished======")

    logging.info("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    logging.info("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def data_preprocessing(x_train, x_test):
    '''
    进去均值除以方差：https://blog.csdn.net/Miss_yuki/article/details/80662017
    :param x_train:
    :param x_test:
    :return:
    '''
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # 有点问题，貌似应该是循例那数据的均值和方差
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def max_pooling(input, k_size=1, stride=1, name=None):
    output = tf.nn.max_pool(input, [1, k_size, k_size, 1], [1, stride, stride, 1],
                            padding='SAME', name=name)
    return output


def conv2d(input, W):
    output = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    return output


def batch_norm(input):
    output = tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                          is_training=train_flag, updates_collections=None)
    return output


def bias_variable(shape):
    bias_var = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return bias_var


def conv1(input):
    # 第一次卷积
    w_conv1_1 = tf.get_variable(name='conv1_1', shape=[3, 3, 3, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = tf.Variable(bias_variable([64]))
    output = tf.nn.relu(batch_norm(conv2d(input, w_conv1_1)) + b_conv1_1)

    w_conv1_2 = tf.get_variable(name='conv1_2', shape=[3, 3, 64, 64],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = tf.Variable(bias_variable([64]))
    output = tf.nn.relu(batch_norm(conv2d(output, w_conv1_2)) + b_conv1_2)
    return output


def conv2(input):
    w_conv2_1 = tf.get_variable(name='conv2_1', shape=[3, 3, 64, 128])
    b_conv2_1 = tf.Variable(bias_variable([128]))
    output = tf.nn.relu(batch_norm(conv2d(input, w_conv2_1)) + b_conv2_1)

    w_conv2_2 = tf.get_variable(name='conv2_2', shape=[3, 3, 128, 128])
    b_conv2_2 = tf.Variable(bias_variable([128]))
    output = tf.nn.relu(batch_norm(conv2d(output, w_conv2_2)) + b_conv2_2)
    return output


def conv3(input):
    w_conv3_1 = tf.get_variable(name='conv3_1', shape=[3, 3, 128, 256])
    b_conv3_1 = tf.Variable(bias_variable([256]))
    output = tf.nn.relu(batch_norm(conv2d(input, w_conv3_1)) + b_conv3_1)

    w_conv3_2 = tf.get_variable(name='conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = tf.Variable(bias_variable([256]))
    output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_2)) + b_conv3_2)

    w_conv3_3 = tf.get_variable(name='conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = tf.Variable(bias_variable([256]))
    output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_3)) + b_conv3_3)

    w_conv3_4 = tf.get_variable(name='conv3_4', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_4 = tf.Variable(bias_variable([256]))
    output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_4)) + b_conv3_4)
    return output


def conv4(input):
    w_conv4_1 = tf.get_variable(name='conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(input, w_conv4_1) + b_conv4_1)

    w_conv4_2 = tf.get_variable(name='conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(output, w_conv4_2) + b_conv4_2)

    w_conv4_3 = tf.get_variable(name='conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(output, w_conv4_3) + b_conv4_3)

    w_conv4_4 = tf.get_variable(name='conv4_4', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_4 = tf.Variable(bias_variable(shape=[512]))
    output = tf.nn.relu(conv2d(output, w_conv4_4) + b_conv4_4)
    return output


def conv5(input):
    w_conv5_1 = tf.get_variable(name='conv5_1', shape=[3, 3, 512, 512])
    b_conv5_1 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(input, w_conv5_1) + b_conv5_1)

    w_conv5_2 = tf.get_variable(name='conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(output, w_conv5_2) + b_conv5_2)

    w_conv5_3 = tf.get_variable(name='conv5_3', shape=[3, 3, 512, 512])
    b_conv5_3 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(output, w_conv5_3) + b_conv5_3)

    w_conv5_4 = tf.get_variable(name='conv5_4', shape=[3, 3, 512, 512])
    b_conv5_4 = tf.Variable(bias_variable([512]))
    output = tf.nn.relu(conv2d(output, w_conv5_4) + b_conv5_4)
    return output


def fc_struc(input, shape, name=None):
    w_fc = tf.get_variable(name=name, shape=shape,
                            initializer=tf.contrib.keras.initializers.he_normal())
    b_fc = tf.Variable(bias_variable([shape[1]]))
    output = tf.nn.relu(batch_norm(tf.matmul(input, w_fc)) + b_fc)
    return output


def learn_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001


def _random_flip_leftright(batch):
    # 随机对图像进程翻转操作
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    # 长宽方向进行增加 纬度方向不变
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            # 增加了图片的尺寸
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                             nw:nw + crop_shape[1]]
    return new_batch


def data_augmentation(batch):
    # 图像增强
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def run_testing(test_x, test_y, sess):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    for i in range(1, 5):
        batch_x = test_x[pre_index:pre_index + batch_size]
        batch_y = test_y[pre_index:pre_index + batch_size]
        pre_index += batch_size
        loss_, acc_ = sess.run([cross_entropy, accuracy],
                             feed_dict={x: batch_x,
                                        y_: batch_y,
                                        keep_prob:1.0,
                                        train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss),
                                tf.Summary.Value(tag='test_acc', simple_value=acc)])
    return acc, loss, summary


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    x = tf.placeholder(tf.float32, [None, image_size, image_size, image_channels])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # 第一次卷积
    logging.info('conv1 start')
    output = conv1(x)
    logging.info('conv1 end')

    # 第一次池化
    logging.info('pool1 start')
    output = max_pooling(output, 2, 2, 'pool1')
    logging.info('pool1 end')

    # 第二次卷积
    logging.info('conv2 start')
    output = conv2(output)
    logging.info('conv2 end')

    # 第二次池化
    logging.info('pool2 start')
    output = max_pooling(output, 2, 2, 'pool2')
    logging.info('pool2 end')

    # 第三次卷积
    logging.info('conv3 start')
    output = conv3(output)
    logging.info('conv3 end')

    # 第三次池化
    logging.info('pool3 start')
    output = max_pooling(output, 2, 2, 'pool3')
    logging.info('pool3 end')

    # 第四次卷积
    logging.info('conv4 start')
    output = conv4(output)
    logging.info('conv4 end')

    # 第四次池化
    logging.info('pool4 start')
    output = max_pooling(output, 2, 2, 'pool4')
    logging.info('pool4 end')

    # 第五次卷积
    logging.info('conv5 start')
    output = conv5(output)
    logging.info('conv5 end')

    # 开始全连接层
    logging.info('fc start')
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    out_fc1 = fc_struc(output, [2048, 4096], 'fc1')
    out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob)
    out_fc2 = fc_struc(out_fc1_drop, [4096, 4096], 'fc2')
    out_fc2_drop = tf.nn.dropout(out_fc2, keep_prob)
    out_fc3 = fc_struc(out_fc2_drop, [4096, 10], 'fc3')
    logging.info('fc end')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_fc3))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_locking=True).\
                 minimize(cross_entropy +  l2 * weight_decay)
    result_predict = tf.equal(tf.argmax(out_fc3, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(result_predict, tf.float32))

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)
        logging.info('start train!')
        for ep_num in range(1, total_epoch + 1):
            lr = learn_rate_schedule(ep_num)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()
            logging.info('epoch: {ep_num}/{total_epoch}'.format(ep_num=ep_num, total_epoch=total_epoch))
            for iter in range(1, iterations + 1):
                batch_x = train_x[pre_index: pre_index + batch_size]
                batch_y = train_y[pre_index: pre_index + batch_size]
                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x,
                                                    y_: batch_y,
                                                    keep_prob: dropout_rate,
                                                    learning_rate: lr,
                                                    train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                pre_index = pre_index + batch_size
                train_acc += batch_acc
                train_loss += batch_loss

                if iter == iterations:
                    avg_acc = train_acc / iterations
                    avg_loss = train_loss / iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=avg_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=avg_acc)])

                    val_acc, val_loss, test_summary = run_testing(test_x, test_y, sess)

                    summary_writer.add_summary(train_summary, ep_num)
                    summary_writer.add_summary(test_summary, ep_num)
                    summary_writer.flush()
                    logging.info("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (iter, iterations, int(time.time() - start_time), avg_loss, avg_acc, val_loss, val_acc))
                else:
                    logging.info("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                          % (iter, iterations, train_loss / iter, train_acc / iter))
        logging.info('end train!')
        save_path = saver.save(sess, model_save_path)
        logging.info("Model saved in file: %s" % save_path)






























