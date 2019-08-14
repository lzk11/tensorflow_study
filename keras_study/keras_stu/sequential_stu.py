from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import  mnist
import numpy


# 第一步选择模型
model = Sequential()


# 第二步构神经网络
# 第一层为输入层
model.add(Dense(500, input_shape=(784, )))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
# 第二层为隐藏层
model.add(Dense(500))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
# 第三层输出层
model.add(Dense(10))
model.add(Activation('softmax'))

# 第三步编译
# 损失函数： https://blog.csdn.net/vhhgfg74466/article/details/87976728
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# 第四步: 进行训练
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == Y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == Y_test[:, None]).astype(int)

model.fit(X_train, Y_train, batch_size=200, epochs=10, shuffle=True, verbose=2, validation_split=0.3)
scores = model.evaluate(X_test, Y_test, batch_size=200, verbose=0)
print("test set loss %.2f" % scores)

result = model.predict(X_test, batch_size=200, verbose=0)
result_max = numpy.argmax(result, axis=1)
test_flag = numpy.argmax(Y_test, axis=1)

result_bool = numpy.equal(result_max, test_flag)

true_num = numpy.sum(result_bool)

print('the acc of test set is %.2f' % (true_num / len(result_bool)))
