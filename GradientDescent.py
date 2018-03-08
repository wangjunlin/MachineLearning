import numpy as np
import matplotlib.pyplot as plt


# 求极小值目标函数
def target_func(x):
    return np.square(x + 1)


# 目标函数一阶导数
def dfunc(x):
    return 2 * (x + 1)


def GradientDescent(x_start, dfunc, epochs, learn_rate=0.001, momentum=0, decay=0):
    '''

    :param x_start: 设置初始坐标
    :param dfunc: 目标函数一阶导数
    :param epochs: 迭代周期
    :param learn_rate: 学习率/步长
    :param momentum: 冲量 当本次梯度下降方向与上次更新量v方向相同，上次更新量能对本次搜索起加速作用
    :param decay: 学习率衰减因子 随着周期增加而减小 0~1
    :return: 每次迭代后的x的位置
    '''
    x_now = np.zeros(epochs + 1)
    x_now[0] = x_start
    v = 0
    for i in range(epochs):
        dx = dfunc(x_start)
        # 加入学习率衰减因子
        learn_decay = learn_rate * 1.0 / (1.0 + decay * i)
        v = -dx * learn_decay + momentum * v  # x改变的幅度,加入冲量加速、减缓v
        x_start += v
        x_now[i + 1] = x_start
    return x_now


def testGradientDescent():
    line_x = np.linspace(-50, 50, 200)
    target_line_x = target_func(line_x)
    x_start = 42
    epochs = 20
    learn_rate = 0.1
    x = GradientDescent(x_start, dfunc, epochs, learn_rate)
    color = 'r'
    plt.plot(line_x, target_line_x, c='b')
    plt.plot(x, target_func(x), c=color, label='lr={}'.format(learn_rate))
    plt.scatter(x, target_func(x), c=color, )
    plt.legend()
    plt.show()
