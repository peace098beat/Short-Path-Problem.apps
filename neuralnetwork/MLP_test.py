# -*- coding: utf-8 -*-
"""

MLP_test.py
多層パーセプトロン
"""

import numpy as np
import sys
from MultiLayerPerceptron import MultiLayerPerceptron
from ndprint import ndprint, ndprints

if __name__ == '__main__':
    # データセットの生成
    # ------------------
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # 多層パーセプトロンの初期化
    mlp = MultiLayerPerceptron(numInput=2, numHidden=5, numOutput=1, activate1="tanh", activate2="identity")
    # パーセプトロンの学習
    mlp.fit(X, y,learning_rate=0.2, epochs=10000)
    # パーセプトロンを実行
    y0 = mlp.predict(X[0])

    for x, y in zip(X, y):
        print 'X:%s, y:%0.2f, pred:%0.2f' % (ndprint(x), y, mlp.predict(x))
