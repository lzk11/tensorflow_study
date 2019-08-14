from keras.layers import  Input
from keras.layers.merge import concatenate
from keras.layers import Dropout, Dense, Activation, Flatten, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model


s = 0
INCEPTION_A_COUNT = 0
INCEPTION_B_COUNT = 0
INCEPTION_C_COUNT = 0


def conv_block(x, num_filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False):
    global  CONV_BLOCK_COUNT
    CONV_BLOCK_COUNT += 1
    with K.name_scope('conv_block_' + str(CONV_BLOCK_COUNT)):
        x =  Conv2D(filters=num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias)(x)
        x = BatchNormalization(axis=-1, momentum=0.997, scale=False)(x)
        x = Activation('relu')(x)
    return x


def stem(input_x):
    with K.name_scope('stem'):
        x_conv1 = conv_block(input_x, 32, (3, 3), (2, 2), 'valid')
        x_conv2 = conv_block(x_conv1, 32, (3, 3), (1, 1), 'valid')
        x_conv3 = conv_block(x_conv2, 64, (3, 3))

        x1_pool1 = MaxPooling2D((3, 3), (2, 2), padding='valid')(x_conv3)
        x1_conv1 = conv_block(x_conv3, 96, (3, 3), (2, 2), padding='valid')
        x1_concat = concatenate([x1_pool1, x1_conv1], axis=-1)

        x2_conv1_1 = conv_block(x1_concat, 64, (1, 1))
        x2_conv1_2 = conv_block(x2_conv1_1, 96, (3, 3), padding='valid')

        x2_conv2_1 = conv_block(x1_concat, 64, (1, 1))
        x2_conv2_2 = conv_block(x2_conv2_1, 64, (7, 1))
        x2_conv2_3 = conv_block(x2_conv2_2, 64, (1, 7))
        x2_conv2_4 = conv_block(x2_conv2_3, 96, (3, 3), padding='valid')
        x2_concat = concatenate([x2_conv1_2, x2_conv2_4], axis=-1)

        x3_conv1 = conv_block(x2_concat, 192, (3, 3), (2, 2), padding='valid')

        x3_conv2 = MaxPooling2D((3, 3), (2, 2), padding='valid')(x2_concat)
        x3_concat = concatenate([x3_conv1, x3_conv2], axis=-1)
    # (35, 35, 394)
    return x3_concat


def inception_A(input_x):
    global INCEPTION_A_COUNT
    INCEPTION_A_COUNT += 1
    with K.name_scope('inception_A_' + str(INCEPTION_A_COUNT)):
        avg_pool1 = AveragePooling2D((3, 3), (1, 1), padding='same')(input_x)
        x_conv1 = conv_block(avg_pool1, 96, (1, 1))

        x_conv2 = conv_block(input_x, 96, (1, 1))

        x_conv3_1 = conv_block(input_x, 64, (1, 1))
        x_conv3_2 = conv_block(x_conv3_1, 96, (3, 3))

        x_conv4_1 = conv_block(input_x, 64, (1, 1))
        x_conv4_2 = conv_block(x_conv4_1, 96, (3, 3))
        x_conv4_3 = conv_block(x_conv4_2, 96, (3, 3))

        x_concat = concatenate([x_conv1, x_conv2, x_conv3_2, x_conv4_3], axis=-1)
    # (35, 35, 384)
    return x_concat


def inception_B(input_x):
    global INCEPTION_B_COUNT
    INCEPTION_B_COUNT += 1
    with K.name_scope('inception_B_' + str(INCEPTION_B_COUNT)):
        avg_pool1 = AveragePooling2D((3, 3), (1, 1), padding='same')(input_x)
        x_conv1 = conv_block(avg_pool1, 128, (1, 1))

        x_conv2 = conv_block(input_x, 384, (1, 1))

        x_conv3_1 = conv_block(input_x, 192, (1, 1))
        x_conv3_2 = conv_block(x_conv3_1, 224, (1, 7))
        x_conv3_3 = conv_block(x_conv3_2, 256, (7, 1))

        x_conv4_1 = conv_block(input_x, 192, (1, 1))
        x_conv4_2 = conv_block(x_conv4_1, 192, (1, 7))
        x_conv4_3 = conv_block(x_conv4_2, 224, (7, 1))
        x_conv4_4 = conv_block(x_conv4_3, 224, (1, 7))
        x_conv4_5 = conv_block(x_conv4_4, 256, (7, 1))

        x_concat = concatenate([x_conv1, x_conv2, x_conv3_3, x_conv4_5], axis=-1)
    return x_concat


def reduction_A(input_x, k=192, l=224, m=256, n=384):
    with K.name_scope('reducion_a'):
        max_pool1 = MaxPooling2D((3, 3), (2, 2), padding='valid')(input_x)

        x_conv2 = conv_block(input_x, n, (3, 3), (2, 2), padding='valid')

        x_conv3_1 = conv_block(input_x, k)
        x_conv3_2 = conv_block(x_conv3_1, l, (3, 3))
        x_conv3_3 = conv_block(x_conv3_2, m, (3, 3), (2, 2), padding='valid')

        x_concat = concatenate([max_pool1, x_conv2, x_conv3_3], axis=-1)
    return x_concat


def reduction_B(inpout_x):
    with K.name_scope('reduction_b'):
        max_pool1 = MaxPooling2D((3, 3), (2, 2), padding='valid')(inpout_x)

        x_conv2_1 = conv_block(inpout_x, 192, (1, 1))
        x_conv2_2 = conv_block(x_conv2_1, 192, (3, 3), (2, 2), padding='valid')

        x_conv3_1 = conv_block(inpout_x, 256, (1, 1))
        x_conv3_2 = conv_block(x_conv3_1, 256, (1, 7))
        x_conv3_3 = conv_block(x_conv3_2, 320, (7, 1))
        x_conv3_4 = conv_block(x_conv3_3, 320, (3, 3), (2, 2), padding='valid')

        x_concat = concatenate([max_pool1, x_conv2_2, x_conv3_4], axis=-1)
    return x_concat


def inception_C(input_x):
    global INCEPTION_C_COUNT
    INCEPTION_C_COUNT += 1
    with K.name_scope('inception_C_' + str(INCEPTION_C_COUNT)):
        avg_pool1 = AveragePooling2D((1, 1), (1, 1), padding='same')(input_x)
        x_conv1 = conv_block(avg_pool1, 256, (1, 1))

        x_conv2 = conv_block(input_x, 256, (1, 1))

        x_conv3_1 = conv_block(input_x, 384, (1, 1))
        x_conv3_2_1 = conv_block(x_conv3_1, 256, (1, 3))
        x_conv3_2_2 = conv_block(x_conv3_1, 256, (3, 1))

        x_conv3_1 = conv_block(input_x, 384, (1, 1))
        x_conv3_2 = conv_block(x_conv3_1, 448, (1, 3))
        x_conv3_3 = conv_block(x_conv3_2, 512, (3, 1))
        x_conv3_4_1 = conv_block(x_conv3_3, 256, (3, 1))
        x_conv3_4_2 = conv_block(x_conv3_3, 256, (1, 3))

        x_concat = concatenate([x_conv1, x_conv2, x_conv3_2_1, x_conv3_2_2, x_conv3_4_1, x_conv3_4_2], axis=-1)
    return x_concat


def inception_v4_backbone(num_classes=1000):
    x_input = Input(shape=(299, 299, 3))

    x = stem(x_input)

    for i in range(4):
        x = inception_A(x)

    x = reduction_A(x, 192, 224, 256, 384)

    for i in range(7):
        x = inception_B(x)

    x = reduction_B(x)

    for i in range(3):
        x = inception_C(x)

    x = AveragePooling2D((8, 8))(x)

    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=x_input, output=x, name='Inception-V4')
    return model


if __name__ == '__main__':
    inception_v4 = inception_v4_backbone()
    plot_model(inception_v4, 'inception_v4_.png', show_shapes=True)

















