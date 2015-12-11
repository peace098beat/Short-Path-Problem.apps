# -*- coding: utf-8 -*-
"""

## Description
    多層パーセプトロンの高速化
    http://aidiary.hatenablog.com/entry/20140201/1391218771
    入力層 - 隠れ層 - 出力層の3層構造で固定（PRMLではこれを2層と呼んでいる）

    隠れ層の活性化関数にはtanh関数またはsigmoid logistic関数が使える
    出力層の活性化関数にはtanh関数、sigmoid logistic関数、恒等関数が使える

## コメント
    前回、3層以上の多層パーセプトロンを実装しようとしたが失敗。
    隠れ層を含んだ、各レイヤーをnumpy配列として、
    一つのリストに格納していたが、どうも型がおかしくなって上手くいかない。
    おそらく、各レイヤーはオブジェクトとして実装したほうがよい。

    今回は、基本に戻り3層パーセプトロンを実装する。
    参考コードは関数化されており、拡張性を秘めているので利用する。

    ただし、拡張性については、拡張するのはいいが、
    いずれは機械学習ライブラリを利用していくほうがよい。

    現状は3層パーセプトロンを実装し、QLearningとの統合を目指す。

(20105/12/10) ver1.1 get_Q_value()を追加
"""
__version__ = "1.1"
import numpy as np
import sys

# numpy配列をプリントした場合の表示桁数
np.set_printoptions(precision=3)


def ndprint(a, format_string='{0:.2f}'):
    """
    ndarrayをprintする関数
    :example: ndprint(x)
    """
    return [format_string.format(v, i) for i, v in enumerate(a)]


def ndprints(s, a, format_string='{0:.2f}'):
    """
    ndarrayをprintする関数
    :example: ndprint(x)
    """
    print s, [format_string.format(v, i) for i, v in enumerate(a)]


############################################
#
#   サブ関数群
#
############################################

def tanh(x):
    """
    tanh(x)
    :param x:
    """
    return np.tanh(x)


def tanh_deriv(x):
    """
    tanhの微分. tanh'(x)
    tanh'(x)=1 - tanh(x)**2
    :param x: tanh(x)
    """
    return 1.0 - x ** 2


def sigmoid(x):
    """
    シグモイド関数
    """
    return 1. / (1. + np.exp(-x))


def sigmoid_deriv(x):
    """
    シグモイド関数の導関数.
    sigmoid'(x) = sigmoid(x) * (1-sigmoid(x))
    :param x: sigmoid(x)
    """
    return x * (1 - x)


def identity(x):
    """
    線形関数.
    """
    return x


def identity_deriv(x):
    """
    線形関数の微分
    :param x:
    """
    return 1


############################################
#
#   多層パーセプトロンクラス
#
############################################

class MultiLayerPerceptron:
    def __init__(self, numInput=3, numHidden=5, numOutput=1, activate1="tanh", activate2="sigmoid"):
        """
        多層パーセプトロンを初期化
        :param numInput: 入力層のユニット数(バイアスユニットは除く)
        :param numHidden: 隠れ層のユニット数(バイアスユニットは除く)
        :param numOutput: 出力層のユニット数
        :param act1: 隠れ層の活性化関数 ( tanh or sigmoid )
        :param act2: 出力層の活性化関数 ( tanh or sigmoid or identity )
        """

        # 引数の指定に合わせて隠れ層の活性化関数とその微分関数を設定
        # ----------------------------------------------------------
        if activate1 == "tanh":
            self.act1 = tanh
            self.act1_deriv = tanh_deriv
        elif activate1 == "sigmoid":
            self.act1 = sigmoid
            self.act1_deriv = sigmoid_deriv
        else:
            print "ERROR: act1 is tanh or sigmoid"
            sys.exit()
        print ">> Hidden Layer Activation Function is : %s" % activate1

        # 引数の指定に合わせて、出力層の活性化関数とその微分関数を設定
        # ----------------------------------------------------------
        if activate2 == "tanh":
            self.act2 = tanh
            self.act2_deriv = tanh_deriv
        elif activate2 == "sigmoid":
            self.act2 = sigmoid
            self.act2_deriv = sigmoid_deriv
        elif activate2 == "identity":
            self.act2 = identity
            self.act2_deriv = identity_deriv
        else:
            print "ERROR: act2 is tanh or sigmoid or identity"
            sys.exit()
        print ">> Outpu Layer Activation Function is : %s" % activate2

        # 各レイヤのユニット数を格納(バイアスユニットがあるので入力層と隠れ層は+1)
        # ------------------------------------------------------------------------
        self.numInput = numInput + 1
        self.numHidden = numHidden + 1
        self.numOutput = numOutput

        # 重みを(-1.0, 1.0)の一様乱数で初期化
        # ------------------------------------
        # -- 入力層 - 隠れ層
        self.weight1 = np.random.uniform(
            -1.0,
            1.0,
            (self.numHidden, self.numInput)
        )
        # -- 隠れ層 - 出力層
        self.weight2 = np.random.uniform(
            -1.0,
            1.0,
            (self.numOutput, self.numHidden)
        )

        # 重みを(-0.00001, 0.00001)のほぼぜろで初期化
        # ! 0ではだめ!学習しない!
        # ------------------------------------
        # -- 入力層 - 隠れ層
        # self.weight1 = np.zeros((self.numHidden, self.numInput))
        # -- 隠れ層 - 出力層
        # self.weight2 = np.zeros((self.numOutput, self.numHidden))

        # print self.weight1
        # print self.weight2

    def fit(self, X, t, learning_rate=0.2, epochs=10000):
        """
        訓練データを用いてネットワークの重みを更新する
        :param X: 入力データ
        :param t: 教師データ
        :param learning_rate: 学習率
        :param epochs: 更新回数
        """
        # バイアスユニットを追加
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # 教師データ
        t = np.array(t)

        err = 0

        # 逐次学習
        # (訓練データからランダムサンプリングして重みを更新。epochs回繰り返す)
        # ---------------------------------------------------------------------
        for k in range(epochs):
            if k % 10 == 0:
                # print '>Epochs:%d' % k
                pass

            # 訓練データからランダムにサンプルを選択
            # --------------------------------------
            i = np.random.randint(X.shape[0])
            x = X[i]


            # 順伝播
            # ======

            # 入力を順伝播させて中間層の出力を計算
            # -------------------------------------
            z = self.act1(np.dot(self.weight1, x))

            # 中間層の出力を順伝播させて出力層の出力を計算
            # --------------------------------------------
            y = self.act2(np.dot(self.weight2, z))


            # 出力層の誤差を計算
            # ==================

            # ** WARNING **
            # PRMLによると出力層の活性化関数にどれを用いても
            # (y - t[i])でよいと書いてあるが
            # 下のように出力層の活性化関数の微分もかけたほうが精度がずっと良くなる

            # 教師データと出力ユニットとの誤差を計算
            # ---------------------------------------
            delta2 = self.act2_deriv(y) * (y - t[i])

            # 出力層の誤差を逆伝播させて隠れ層の誤差を計算
            # -------------------------------------------
            delta1 = self.act1_deriv(z) * np.dot(self.weight2.T, delta2)


            # 重みの更新
            # ==========

            # (行列演算になるので2次元ベクトルに変換する必要がある)
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)

            # 隠れ層の誤差を用いて隠れ層の重みを更新
            # -----------------------------------------------------
            # self.weight1 = self.weight1 - learning_rate * np.dot(delta1.T, x)
            self.weight1 -= learning_rate * np.dot(delta1.T, x)

            # (行列演算になるので2次元ベクトルに変換する必要がある)
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)

            # 出力層の誤差を用いて、出力層の重みを更新
            # -----------------------------------------
            # self.weight2 = self.weight2 - learning_rate * np.dot(delta2.T, z)
            self.weight2 -= learning_rate * np.dot(delta2.T, z)

            err += np.sum(delta2)
        return err
    def predict(self, x0):
        """
        テストデータの出力を予測し、教師と比較
        :param x:
        """
        x = np.array(x0)

        # バイアスの1を追加
        x = np.insert(x, 0, 1)

        # 順伝播によりネットワークの出力を計算
        z = self.act1(np.dot(self.weight1, x))
        y = self.act2(np.dot(self.weight2, z))

        return y



if __name__ == '__main__':
    mpl = MultiLayerPerceptron(numInput=2, numHidden=5, numOutput=1, activate1="tanh", activate2="sigmoid")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    err = mpl.fit(X, y)
    print(err)
    for x, y in zip(X, y):
        print 'X:%s, y:%0.2f, pred:%0.2f' % (ndprint(x), y, mpl.predict(x))
