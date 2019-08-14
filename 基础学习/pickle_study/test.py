# pickle库将模型参数保存到本地磁盘，以备在后续的预测阶段加载模型
# dump() 将变量内容保存到文件中
# load() 从dump的文件加载数据到变量中
# dumps() 将变量内容变为字符串存储，并未加载到内存
# loads() 将dumps的结果变为真是的内容
# loadl可以以dump的顺序不断读取内容
import pickle

dataList = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
dataDic = { 0: [1, 2, 3, 4],
            1: ('a', 'b'),
            2: {'c':'yes','d':'no'}}

file = open('./datafile.txt', 'wb')
pickle.dump(dataList, file)
pickle.dump(dataDic, file)
file.close()

file = open('./datafile.txt', 'rb')
dataL = pickle.load(file)
print(dataL)
dataD = pickle.load(file)
print(dataD)



# dumps将需要打包
p = pickle.dumps(dataList)
print(pickle.loads(p))
p = pickle.dumps(dataDic)
print(pickle.loads(p))