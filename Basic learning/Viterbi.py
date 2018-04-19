from numpy import *


# '''
# 为了简化算法，防止矩阵增大使计算量复杂，使用Viterbi算法实现HMM模型预测天气，参数同HMM.py
# '''


def viterbi(obvious, states, start_P, trans_P, emit_P):
    """
    :param obvious: 观测序列 {干旱，干燥，湿润，潮湿}
    :param states: 隐状态 {晴天，阴天，雨天}
    :param start_P: 起始概率（隐状态）
    :param trans_P: 转移概率
    :param emit_P: 发射概率（隐状态转为显状态的概率）
    """
    V = [{}]  # 路径概率表，即所有序列中在T时刻以状态I终止时最大概率
    for y in states:  # 初始化隐状态的初始状态 (t=0)
        V[0][y] = start_P[y] * emit_P[y][obvious[0]]
    for t in range(1, len(obvious)):  # 对t > 0执行一次
        V.append({})
        for y in states:
            # 隐状态 = 前一个状态是y0的概率 * y0 转移到 y 的概率 * y 表现为当前状态的概率
            V[t][y] = max([(V[t - 1][y0] * trans_P[y0][y] * emit_P[y][obvious[t]]) for y0 in states])
    result = []
    for vector in V:
        temp = {list(vector.keys())[argmax(list(vector.values()))]: max(vector.values())}
        result.append(temp)
    return result


obvious = ('干旱', '干燥', '潮湿')
states = ('晴天', '阴天', '雨天')
start_P = {'晴天': 0.63, '阴天': 0.17, '雨天': 0.20}
trans_P = {'晴天': {'晴天': 0.50, '阴天': 0.375, '雨天': 0.125},
           '阴天': {'晴天': 0.25, '阴天': 0.125, '雨天': 0.625},
           '雨天': {'晴天': 0.25, '阴天': 0.375, '雨天': 0.375}}
emit_P = {'晴天': {'干旱': 0.60, '干燥': 0.20, '湿润': 0.15, '潮湿': 0.05},
          '阴天': {'干旱': 0.25, '干燥': 0.25, '湿润': 0.25, '潮湿': 0.25},
          '雨天': {'干旱': 0.05, '干燥': 0.10, '湿润': 0.35, '潮湿': 0.50}}
print(viterbi(obvious, states, start_P, trans_P, emit_P))
