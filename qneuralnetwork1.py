# coding: utf-8
"""
qneuralnetwork1.py
"""

import numpy as np

# from QLearning.QLearn import QLearning, Agent, State
from qlearning import QLearn
from neuralnetwork.MultiLayerPerceptron import MultiLayerPerceptron, ndprint



# ログ用の定数
goaled_number = 0
MAX_ITERATE = 10
GOAL = QLearn.GOAL

agent = QLearn.Agent()
state = QLearn.State()


# 初期設定
S = state.getInitState()
print '>> State : Init State'
print S

mpl = MultiLayerPerceptron(numInput=2, numHidden=5, numOutput=4, activate1="tanh", activate2="sigmoid")
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])
# mpl.fit(X, y)

# for x, y in zip(X, y):
#     print 'X:%s, y:%0.2f, pred:%0.2f' % (ndprint(x), y, mpl.predict(x))
# X = np.array([0,0])
# Y = np.array([0,0,0,0])
X = []
Y = []
for i in range(MAX_ITERATE):
    # if i % (MAX_ITERATE / 20) == 0:
    print '--------------------------'
    print 'Loop : ',i
        # agent.displayQ()

    # 1. エージェントは環境から受け取った観測Sを受け取り、方策planに基づいて環境に行動aを渡す
    # a = agent.getNextAction(np.copy(S))
    # print '>>Agent Next Action is :%d' % a
    o = state.encodeStateToO(np.copy(S))

    # 観測(observe)から、NNでQ値(配列)を取得
    Qt = mpl.predict(o)
    print Qt
    maxQt_idx = np.argmax(Qt)
    a = maxQt_idx

    # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'を返却
    S_next, r, option = state.getNextState(np.copy(S), a)
    # print '>>State Next step:'
    # print S_next
    # print '>>> Reward ', r
    # print '>>> Option ', option
    o_next = state.encodeStateToO(np.copy(S_next))

    # 3. Agentに学習させる
    # agent.learn(np.copy(S), a, r, np.copy(S_next))
    D = [o, a, r, o_next]
    print 'D:',D

    # 4. NNでQ値を算出
    Q = mpl.predict(o)
    ALPHA = 0.9
    GAMMA = 0.9
    qk = Q[a]
    maxQ = np.max(Q)
    Q[a] = qk + ALPHA * (r + GAMMA * maxQ - qk)
    print 'Qs:',Q


    # 3. NeauralNetworkで学習する
    X.append(np.array(o))
    Y.append(Q)
    if len(X) > 1000:
        X.pop(0)
        Y.pop(0)

    mpl.fit(np.asarray(X), np.asarray(Y))


    # 4. Stateの初期化判定
    if option == GOAL:
        goaled_number += 1
        S = state.getInitState()
    else:
        S = S_next
    print '>> GOAL NUMBER :', goaled_number


print '>> GOAL NUMBER :', goaled_number
# print agent.displayQ()

