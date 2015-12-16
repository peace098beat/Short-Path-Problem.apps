# coding: utf-8
"""
qneuralnetwork3.py

参考
Q学習-最良経路を学習するスクリプト書いた (powered by Python)
http://d.hatena.ne.jp/Kshi_Kshi/20111227/1324993576
Greedy法:学習結果をGreedy法で行動選択
"""

import numpy as np
from qlearning import QLearn
from neuralnetwork.MultiLayerPerceptron import MultiLayerPerceptron, ndprint

import sys
import copy
import random
import numpy as np

# PySide系モジュール
from PySide.QtGui import *
from PySide.QtCore import *

# 解析条件
GAMMA = 0.9
ALPHA = 0.2 #上から0.5, 0.2,0.2
GREEDY_RATIO = 0.5

# MAX_ITERATE = 50000

# フィールドサイズ
# NUM_COL = 15 + 2  # 横
# NUM_ROW = 15 + 2  # 縦

# 報酬
GOAL_REWORD = 1

# 行動パタン
ACTION = [0, 1, 2, 3]
ACTION_NAME = ['UP', 'RIGHT', 'DOWN', 'LEFT']
NUM_ACTION = 4

# 道:0 壁:1 ゴール:2 エージェント:3
ROAD, WALL, GOAL, AGENT = 0, 1, 2, 3

RANDOM_WALL_RATIO = 0.01

# デバッグプリント
DEBUG_PRINT = True


class Field(object):
    def __init__(self, nrow, ncol):
        # 初期位置とゴール位置
        self.START_ROW, self.START_COL = 1, 1
        self.GOAL_ROW, self.GOAL_COL = nrow - 2, ncol - 2
        # フィールドを生成
        field = np.zeros((nrow, ncol))
        for row in range(nrow):
            for col in range(ncol):
                if row in (0, nrow - 1):
                    field[row, col] = WALL
                if col in (0, ncol - 1):
                    field[row, col] = WALL

        field[self.START_ROW, self.START_COL] = AGENT
        field[self.GOAL_ROW, self.GOAL_COL] = GOAL

        area = nrow * ncol
        for i in range(int(area * RANDOM_WALL_RATIO)):
            row = random.randint(2, nrow - 3)
            col = random.randint(2, ncol - 3)
            field[row, col] = WALL
        self.field = field

    def getField(self):
        return self.field

    def getGoalAxis(self):
        """
        ゴールの座標値を返す
        :return:
        """
        return self.GOAL_ROW, self.GOAL_COL


def _debugPrint(s):
    if DEBUG_PRINT:
        print s


def fieldDisplay(S):
    print '*** Field ***'
    Sprint = np.copy(S)
    Sprint[GOAL_ROW, GOAL_COL] = GOAL
    for row in range(NUM_ROW):
        print Sprint[row, :]


# Agentクラス
class Agent(object):
    """ ゲームルールによらない汎用性を持たす
    action: パターンの数だけ保持
    学習アルゴリズム: Q学習
    a = getNextAction(s)
    lean(S,a,r,S_next)
    """

    def __init__(self, numAction=4):
        self.action_paturn = range(numAction)
        self.learningObj = MultiLayerPerceptron(numInput=2, numHidden=5, numOutput=4, activate1="tanh",
                                                activate2="sigmoid")
        self.X = []
        self.Y = []
        self.learnFlg = True

    def displayQ(self):
        self.learningObj.displayQ()

    def setLearnFlg(self, b):
        self.learnFlg = b

    def learn(self, o, a, r, o_next):
        """Q学習 or NeuralNetworkを使って,Q値を学習"""
        dQs = self.learningObj.predict(o)
        qk = dQs[a]
        maxQ = np.max(dQs)
        dQs[a] = qk + ALPHA * (r + GAMMA * maxQ - qk)

        self.X.append(np.asarray(o))
        self.Y.append(np.asarray(dQs))

        if len(self.X) > 500:
            self.X.pop(0)
            self.Y.pop(0)

        err = self.learningObj.fit(np.copy(self.X), np.copy(self.Y), learning_rate=0.2, epochs=500)
        return err

    def getNextAction(self, o):
        Agent_row = o[0]
        Agent_col = o[1]

        # 最大Q値の行動選択, 観測(observe)から、NNでQ値(配列)を取得
        Q_t = self.learningObj.predict(o)

        best_actions = []
        max_Q = -1000000
        for i in range(len(Q_t)):
            q = Q_t[i]
            if q > max_Q:
                max_Q = q
                best_actions = [ACTION[i]]
            elif q == max_Q:
                best_actions.append(ACTION[i])
        # 行動選択(複数ある場合に選ぶ)
        a = np.random.choice(best_actions)

        # 非学習
        if not self.learnFlg:
            return a

        # 学習中
        # greedyの行動選択
        if GREEDY_RATIO < random.random():
            return a
        else:
            return np.random.choice([0, 1, 2, 3])

    def getMaxQvalue(self, o):
        return np.max(self.learningObj.predict(o))

    def get_Q_values(self, o):
        return self.learningObj.predict(o)


# 環境クラス
class State(object):
    """ゲームの中身
    S_NEXT, R = goNextStep(S,a)
    """

    def __init__(self, field):
        self.field = field
        self.initS = self.field.getField()

    def __getReward(self, Snext):
        # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'に変化
        row1, col1 = self.encodeStateToO(Snext)

        # GOALの場合は,報酬を返却
        GOAL_ROW, GOAL_COL = self.field.getGoalAxis()
        if row1 == GOAL_ROW and col1 == GOAL_COL:
            r = GOAL_REWORD
        else:
            r = 0

        return r

    def encodeStateToO(self, S):
        """環境をField全体から、Agentの座標だけに変換"""
        Agent_row, Agent_col = np.where(S == AGENT)
        row, col = Agent_row[0], Agent_col[0]
        return [row, col]

    def getNextState(self, Snow, a):
        """
        Agentの行動と環境から、次の状態と行動を返す
        :return: S_next, reward, option
        """

        # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'に変化
        row1, col1 = self.encodeStateToO(Snow)

        # 行動により状態遷移
        if a == ACTION[0]:
            row2 = row1 - 1
            col2 = col1
        elif a == ACTION[1]:
            row2 = row1
            col2 = col1 + 1
        elif a == ACTION[2]:
            row2 = row1 + 1
            col2 = col1
        elif a == ACTION[3]:
            row2 = row1
            col2 = col1 - 1

        # 状態更新用に今の状態を保持
        Stmp = Snow[row2, col2]

        # 更新
        if Stmp == WALL:
            # 壁判定
            S_next = np.copy(Snow)
        else:
            # Agentを進行して道で上書き
            Snow[row1, col1] = ROAD
            # 道をAgentで上書き
            Snow[row2, col2] = AGENT
            # 状態を更新
            S_next = np.copy(Snow)

        # クラスの外側で、状態判定をする為にOptionを返す
        GOAL_ROW, GOAL_COL = self.field.getGoalAxis()
        if row2 == GOAL_ROW and col2 == GOAL_COL:
            option = GOAL
        else:
            option = ROAD

        return S_next, self.__getReward(S_next), option

    def getInitState(self):
        return self.initS


# 描画用PySideクラス
class GameWindow(QWidget):
    # 定数
    # width = NUM_COL
    # height = NUM_ROW
    # 画面サイズにより柔軟に変更すべき
    interval_time = 1

    def __init__(self, parent=None, nrow=5, ncol=5):
        QWidget.__init__(self, parent)
        self.width = ncol
        self.height = nrow
        LENGTH = 300
        self.setFixedSize(LENGTH, LENGTH)
        self.Margin = LENGTH / self.width

        # クラスプロパティ
        # -----------------
        # 反復数
        self.iter_num = 0

        # ログ用の定数
        self.goaled_number = 0

        self.learn_flg = True

        # 画面バッファ
        self.pixmap = QPixmap(self.size())

        # 解析用オブジェクト
        self.agent = Agent(numAction=4)
        self.field = Field(nrow, ncol)
        self.state = State(self.field)
        self.S = self.state.getInitState()  # プロパティ化

        # 初期画面の準備
        # --------------
        # 画面バッファの初期化
        self.refreshPixmap()
        # グリッドの表示
        painter = QPainter(self.pixmap)
        self.drawGrid(painter)

        # メインループの準備と開始
        # -------------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.mainloop)
        self.timer.start(self.interval_time)

    # ************************************************************* #
    # メインループ
    # ************************************************************* #
    def mainloop(self):
        """
        アニメーションのメインループ
        アルゴリズムの時間更新等はここで行う
        """
        self.iter_num += 1

        # 学習状態を設定 True:学習中, False:非学習
        # self.agent.setLearnFlg(self.learn_flg)

        # 初期設定
        # -----------------------------
        S = self.S
        o = self.state.encodeStateToO(S)

        # 1. エージェントは環境から受け取った観測Sを受け取り、
        # 方策planに基づいて環境に行動aを渡す
        # -----------------------------
        a = self.agent.getNextAction(np.copy(o))

        # 2. 環境StateはエージェントAgentから受け取った行動aと、
        # 現在の状態Sにもとづいて、次の状態S'を返却
        # -----------------------------
        S_next, r, option = self.state.getNextState(np.copy(S), a)
        o_next = self.state.encodeStateToO(np.copy(S_next))

        # 3. Agentに学習させる
        # -----------------------------
        if self.learn_flg:
            self.err = self.agent.learn(np.copy(o), a, r, np.copy(o_next))
            # print('err', self.err)

        # 4. Stateの初期化判定
        # -----------------------------
        if option == GOAL:
            self.goaled_number += 1
            S = self.state.getInitState()
        else:
            S = S_next

        # プロパティの格納
        # -----------------------------
        self.S = np.copy(S)
        self.cursor_pos = np.copy(o_next)
        self.a = a

        # 描画準備
        # -----------------------------
        painter = QPainter(self.pixmap)

        # 画面更新
        self.update()

    def paintEvent(self, *args, **kwargs):
        # QPainterを生成
        painter = QStylePainter(self)
        # QPainterでバッファに準備したデータを描画
        painter.drawPixmap(0, 0, self.pixmap)
        self.drawGrid(painter)
        # エージェントの位置をバッファに書き込み
        self.drawState(painter)
        # デバッグ文字をバッファに書き込み
        self.drawDebugLog(painter)

    # ************************************************************* #
    # 描画系補助関数
    # ************************************************************* #
    def drawDebugLog(self, painter):
        # -- 描画位置
        x = 0.1 * self.size().width()
        y = 0.2 * self.size().height()
        dx = 0.8 * self.size().width()
        dy = 0.3 * self.size().height()
        # x,y,dx,dy = 10,10,1000,1000
        # -- 描画文字
        label = 'Goal:%d / Iter:%d\n' % (self.goaled_number, self.iter_num)

        if self.learn_flg:
            label_learn = 'Traning...%0.5f\n' % np.abs(self.err)
            label += label_learn

        # 行動の表示
        label_action = 'action :no %d, %s\n' % (self.a, ACTION_NAME[self.a])
        label += label_action

        painter.setFont(QFont('Helvetica [Cronyx]', 13, QFont.Bold))
        painter.drawText(x, y, dx, dy, Qt.AlignLeft | Qt.AlignTop, unicode('%s' % label))

    def drawState(self, painter):
        # Cursorの描画
        dx, dy = self.Margin, self.Margin
        painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
        painter.drawRect(self.cursor_pos[0] * dx, self.cursor_pos[1] * dy, dx, dy)

    def drawGrid(self, painter):
        """ マップを表示する関数
         (AgentやGoal,Startはメイン内で描画する"""
        dx = self.Margin
        dy = self.Margin
        # print self.field
        for row in range(self.height):
            for col in range(self.width):
                x, y = row, col
                if self.field.getField()[x][y] == WALL:
                    painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
                elif self.field.getField()[x][y] == ROAD:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))

                    _o = [x, y]
                    qval = self.agent.getMaxQvalue(_o) / 1 * 255.
                    # print(self.agent.get_Q_values(_o))

                    if qval != 0:
                        painter.setBrush(QBrush(QColor(qval, 255 - qval, 255 - qval, 255), Qt.SolidPattern))
                elif self.field.getField()[x][y] == GOAL:
                    painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                else:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))

                painter.drawRect(x * dx, y * dy, dx, dy)

    # ************************************************************* #
    # その他Qt関連補助関数
    # ************************************************************* #
    def refreshPixmap(self):
        """
        画面バッファの初期化関数
        """
        # 画面バッファの初期化
        self.pixmap = QPixmap(self.size())
        # 画面を塗りつぶし (おまじない)
        self.pixmap.fill(self, 0, 0)
        self.pixmap.fill(Qt.white)
        # ぺインターの生成 (おまじない)
        painter = QPainter(self.pixmap)
        # ぺインターによる初期化 (おまじない)
        painter.initFrom(self)
        pass

    def sizeHint(self):
        return QSize(self.width * self.Margin + self.Margin, self.height * self.Margin + self.Margin)

    def keyPressEvent(self, event):
        """子供ウィジェットの場合は、無効"""
        e = event.key()

        if e == Qt.Key_Up:
            self.cursor_pos[1] -= 1
            if self.cursor_pos[1] < 0:
                self.cursor_pos[1] = 0

        elif e == Qt.Key_Down:
            self.cursor_pos[1] += 1
            if self.cursor_pos[1] > NUM_COL - 1:
                self.cursor_pos[1] = NUM_COL - 1

        elif e == Qt.Key_Left:
            self.cursor_pos[0] -= 1
            if self.cursor_pos[0] < 0:
                self.cursor_pos[0] = 0

        elif e == Qt.Key_Right:
            self.cursor_pos[0] += 1
            if self.cursor_pos[0] > NUM_ROW - 1:
                self.cursor_pos[0] = NUM_ROW - 1

        elif e == Qt.Key_Plus:
            self.interval_time -= 20
            if self.interval_time < 1:
                self.interval_time = 1
            self.timer.setInterval(self.interval_time)
            # self.timer.setInterval(1000)
        elif e == Qt.Key_Minus:
            self.interval_time += 20
            if self.interval_time > 200:
                self.interval_time = 200

            self.timer.setInterval(self.interval_time)
            # self.timer.setInterval(1)
        elif e == Qt.Key_Q:
            self.close()
        elif e == Qt.Key_L:
            if self.learn_flg:
                self.learn_flg = False
                print 'LEARN FLG:', False
            else:
                self.learn_flg = True
                print 'LEARN FLG:', True
        else:
            pass

        print 'INTERVAL_TIME', self.interval_time
        self.update()

    pass


class GameMainWindow(QMainWindow):
    """
    シミュレーション用のメインウィンドウ
    各シミュレータのwidgetを描画。
    設定用のコントロールUIを設置。
    """
    WIDTH = 600
    HEIGHT =300

    def __init__(self):
        QMainWindow.__init__(self)
        self.setGeometry(100, 250, self.WIDTH, self.HEIGHT)

        # メインウィンドウのセントラルウィジェット
        self.centralwidget = QWidget(self)
        self.centralwidget.resize(self.size())

        # 操作用Radioボタン
        self.radioButton = QRadioButton('Traning', self.centralwidget)
        self.radioButton.setGeometry(QRect(20, 20, 90, 16))
        self.radioButton.move(50, 50)
        self.radioButton.setChecked(True)

        # 操作用のプッシュボタン
        # self.pushButton = QPushButton(self.centralwidget)
        # self.pushButton.setObjectName("pushButton")
        # self.pushButton.setGeometry(QRect(30, 0, 75, 23))

        # シミュレーション描画ウィンドウ(右上)
        self.widget1 = GameWindow(self.centralwidget, nrow=8, ncol=8)
        self.widget1.setGeometry(QRect(300, 0, 300, 300))
        self.widget1.setFixedSize(300, 300)

        # # シミュレーション描画ウィンドウ(左下)
        # self.widget2 = GameWindow(self.centralwidget, ncol=15, nrow=15)
        # self.widget2.setGeometry(QRect(0, 300, 300, 300))
        # self.widget2.setFixedSize(300, 300)
        #
        # # シミュレーション描画ウィンドウ(右下)
        # self.widget3 = GameWindow(self.centralwidget, ncol=20, nrow=20)
        # self.widget3.setGeometry(QRect(300, 300, 300, 300))
        # self.widget3.setFixedSize(300, 300)

        # self.pushButton.clicked.connect(self.pushedButton)
        self.radioButton.toggled.connect(self.pushRadio)

    @Slot()
    @Slot(bool)
    def pushRadio(self, b=None):
        self.widget1.learn_flg = b
        # self.widget2.learn_flg = b
        # self.widget3.learn_flg = b

    def keyPressEvent(self, event):
        self.widget1.keyPressEvent(event)
        e = event.key()
        if e == Qt.Key_Q:
            self.close()
        elif e == Qt.Key_L:
            if self.learn_flg:
                self.learn_flg = False
                print 'LEARN FLG:', False
            else:
                self.learn_flg = True
                print 'LEARN FLG:', True
        else:
            pass


def main2():
    app = QApplication(sys.argv)
    # win = GameWindow()
    # win.show()
    mwin = GameMainWindow()
    mwin.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main2()
