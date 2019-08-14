from keras import backend as K
# 序贯模型：单输入单输出，一条路到底，层与层之间只有相邻关系，不能跨层连接，编译速度较快，操作简单
from keras.models import Sequential
# 函数时模型： 多输入多输出，层与层之间任意连接，编译速度较慢
from keras.models import Model
from keras.layers.core import Lambda, Dropout, Dense, Activation
import numpy as np


def sub_mean(x):
    x -= K.mean(x, axis=1, keepdims=True)
    return x

def get_submean_model():
    model = Sequential()
    model.add(Dense(5, input_dim=7))
    model.add(Lambda(sub_mean, output_shape=lambda input_shape:input_shape))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

model = get_submean_model()
res = model.predict(np.random.random((3, 7)))
print(res)

