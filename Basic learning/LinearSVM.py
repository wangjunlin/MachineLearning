import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# '''
# 本例将学习线性支持向量机对鸢尾花数据集分类，判断是否为山鸢尾花
# 这里损失函数引入soft margin损失函数：http://blog.csdn.net/jackie_zhu/article/details/52097306
# '''

sess = tf.Session()

iris = datasets.load_iris()
# print(iris)

x_vals = np.array([[x[0], x[3]] for x in iris.data])  # 花萼长和宽
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])  # 0为山鸢尾，把山鸢尾设为1其他为-1

# 分割数据集为训练和测试
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)  # 随机选择80%的数据
test_indices = np.array(list(set(range(len(x_vals))) - set(range(len(train_indices)))))  # 测试集为总数减去训练集的数量
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 100

x_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])  # n*2的矩阵,即花萼长和宽
target_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # n*1 ， 1与-1
A = tf.Variable(tf.random_normal(shape=[2, 1]))  # n*2矩阵乘以2*1矩阵，得到结果为n*1的矩阵即为输出的shape
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# 声明模型输出
model_output = tf.subtract(tf.matmul(x_data, A), b)  # y = Ax - b
# 声明最大间隔损失函数:(1/n)*max(0,1-yn(Axn-b))+alpha*(l2_norm)^2
l2_norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, target_y))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
# 定义预测函数和准确度函数
prediction = tf.sign(model_output)  # 返回函数符号 -1or0or1
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, target_y), dtype=tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)
sess.run(tf.global_variables_initializer())

loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, target_y: rand_y})
    # 记录损失函数
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, target_y: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, target_y: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, target_y: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)
    # 每100次迭代输出当前的损失函数
    if (i + 1) % 100 == 0:
        print('Loss = {} ,A = {} ,b = {}'.format(temp_loss, sess.run(A), sess.run(b)))

[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2 / a1
y_intercept = b / a1

x1_vals = [d[1] for d in x_vals]

best_fit = []  # 分割线的点集合
for i in x1_vals:
    best_fit.append(slope * i + y_intercept)  # y = slope * i + b

# 取出山鸢尾花的横纵坐标
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]

not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]
# 绘制鸢尾花点
plt.plot(setosa_x, setosa_y, 'o', label='setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label="not-setosa")

plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim(0, 10)
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal width')
plt.xlabel('width')
plt.ylabel('length')
plt.show()
