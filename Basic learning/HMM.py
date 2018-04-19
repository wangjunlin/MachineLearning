from numpy import *

# '''
# HMM模型实例:预测在独立空间中通过湿度计来观测隐状态：天气的变化
# 隐状态集合{晴天，阴天，雨天}  观测状态集合{干旱，干燥，湿润，潮湿}
# '''

# 已知状态参数
startP = mat([0.63, 0.17, 0.20])  # 晴天、阴天、雨天起始概率
# 状态转移矩阵
stateP = mat([[0.5, 0.375, 0.125],  # T\T+1
              [0.25, 0.125, 0.625],
              [0.25, 0.375, 0.375]])
# 发射概率矩阵
emitP = mat([[0.60, 0.20, 0.15, 0.05],  # 晴天: 干旱，干燥，湿润，潮湿
             [0.25, 0.25, 0.25, 0.25],  # 阴天
             [0.05, 0.10, 0.35, 0.50]])  # 雨天
#  计算连续3天概率： 观测到{干旱-干燥-潮湿} 时出现天气的最大概率
state1Emit = multiply(startP, emitP[:, 0].T)  # a1(j) = Pi(j)*Bjk1
print('state1Emit : {} ,maxP : {}'.format(state1Emit, state1Emit.argmax()))

state2Emit = stateP.T * state1Emit.T  # 状态转移概率计算 P(晴天*晴天+晴天*阴天+晴天*雨天)
state2Emit = multiply(state2Emit, emitP[:, 1]).T
print('state2Emit : {} ,maxP : {}'.format(state2Emit, state2Emit.argmax()))

state3Emit = stateP.T * state2Emit.T
state3Emit = multiply(state3Emit, emitP[:, 2]).T
print('state3Emit : {} ,maxP : {}'.format(state3Emit, state3Emit.argmax()))
