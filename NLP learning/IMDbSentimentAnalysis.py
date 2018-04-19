from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras import losses
from keras import optimizers
from keras.layers.recurrent import LSTM
import LoadIMDb

# 具体api可以看keras中文文档http://keras-cn.readthedocs.io/en/latest/
# 基于多层感知机MLP实现的基于IMDb影评的情感分析 简单小DEMO

# 建立线性堆叠模型，详细见keras文档
model = Sequential()
# 建立嵌入层 数字列表->向量列表
output_dim = 32
input_dim = 2000
input_length = 100
model.add(Embedding(output_dim=output_dim, input_dim=input_dim, input_length=input_length))
model.add(Dropout(0.2))
#  建立多层感知器模型
model.add(Flatten())  # 把多维输入一维化
# model.add(LSTM(32))  # 加入lstm，要把model.add(Flatten())注释掉
model.add(Dense(units=256, activation='relu'))  # 激活函数设为ReLU
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))  # 该层只有1个神经元，输出1为正评
print('look up model summary:'.format(model.summary()))
# 设置损失函数
# optimizer = optimizers.SGD(lr=0.1, momentum=0.5)
model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
# 开始训练
x_train, y_train, x_test, y_test, train_text, test_text = LoadIMDb.text2dict()
batch_size = 100
epochs = 20
verbose = 2  # 每个epoch输出一行记录
# validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
train_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                          validation_split=0.2)

# 评估模型准确率
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)

# 利用模型进行预测
predict = model.predict_classes(x_test)
predict_result = predict.reshape(-1)  # 转为1维数组便于查看
print('look up 0->49 predict result: {}'.format(predict_result[:50]))


# 将评价数字转为词语展示
def display_Sentiment(index):
    SentimentDict = {1: '好评', 0: '差评'}
    print('原评论内容: {},原label的值: {},\t预测结果: {}'.format(test_text[index], SentimentDict[y_test[index]],
                                                     SentimentDict[predict_result[index]]))


display_Sentiment(5)
