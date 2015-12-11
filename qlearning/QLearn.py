# coding: utf-8
"""
QLearn.py
QLearn3.pyの改良版。
AgentクラスとStateクラスとQLearnクラスを実装
Qtでアニメーションを実装


参考
Q学習-最良経路を学習するスクリプト書いた (powered by Python)
http://d.hatena.ne.jp/Kshi_Kshi/20111227/1324993576
Greedy法:学習結果をGreedy法で行動選択
"""

import sys
import copy
import random
import numpy as np

# PySide系モジュール
from PySide.QtGui import *
from PySide.QtCore import *

# 環境: State
# 環境の状態: S
# エージェントの行動:a
# 方策: Plan -> a = Plan(S)
# S' = State(S,a)
# 報酬: r = Reward(S')

# 解析条件
GAMMA = 0.9
ALPHA = 0.4
GREEDY_RATIO = 0.5

MAX_ITERATE = 50000

# フィールドサイズ
NUM_COL = 15 + 2  # 横
NUM_ROW = 15 + 2  # 縦

# 報酬
GOAL_REWORD = 1

# 行動パタン
ACTION = [0, 1, 2, 3]
ACTION_NAME = ['UP', 'RIGHT', 'DOWN', 'LEFT']
NUM_ACTION = 4

# 初期位置とゴール位置
START_ROW, START_COL = 1, 1
GOAL_ROW, GOAL_COL = NUM_COL - 2, NUM_ROW - 2

# 道:0 壁:1 ゴール:2 エージェント:3
ROAD, WALL, GOAL, AGENT = 0, 1, 2, 3

# フィールドを生成
FIELD = np.zeros((NUM_ROW, NUM_COL))
for row in range(NUM_ROW):
    for col in range(NUM_COL):
        if row in (0, NUM_ROW - 1):
            FIELD[row, col] = WALL
        if col in (0, NUM_COL - 1):
            FIELD[row, col] = WALL
FIELD[START_ROW, START_COL] = AGENT
FIELD[GOAL_ROW, GOAL_COL] = GOAL


def fieldDisplay(S):
    print '*** Field ***'
    Sprint = np.copy(S)
    Sprint[GOAL_ROW, GOAL_COL] = GOAL
    for row in range(NUM_ROW):
        print Sprint[row, :]


# Agentの学習器
# のにちQ-Networkへ変更予定
class QLearning(object):
    """
    Agentがもつ学習クラス。
    a = getAction(S)
    lean(S,a,r,S_next)
    """

    def __init__(self):
        # テーブル関数を準備
        self.Q = np.zeros((NUM_ROW, NUM_COL, NUM_ACTION))
        pass

    def getQ(self):
        return self.Q

    def qlearn(self, S, a, r, S_next):
        row, col = np.where(S == AGENT)
        row = row[0]
        col = col[0]

        row_next, col_next = np.where(S_next == AGENT)
        row_next = row_next[0]
        col_next = col_next[0]

        max_Q = max(self.Q[row_next, col_next, :])
        q = self.Q[row, col, a] + ALPHA * (r + GAMMA * max_Q - self.Q[row, col, a])
        self.Q[row, col, a] = q

        if r == GOAL_REWORD:
            self.Q[row, col, a] = r

    def displayQ(self):
        Q = self.Q
        for row in range(NUM_ROW):
            c = [max(Q[row, col, :]) for col in range(NUM_COL)]
            # print '%5.1f,'*6 % tuple(c)
        for a in ACTION:
            print 'action', a
            for row in range(NUM_ROW):
                c = [Q[row, col, a] for col in range(NUM_COL)]
                print '%4.1f,' * NUM_COL % tuple(c)


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
        self.qlearnobj = QLearning()
        pass

    def displayQ(self):
        self.qlearnobj.displayQ()

    def learn(self, S, a, r, S_next):
        """Q学習 or NeuralNetworkを使って,Q値を学習"""
        self.qlearnobj.qlearn(S, a, r, S_next)

    def getNextAction(self, S):
        Q = self.qlearnobj.getQ()
        Agent_row, Agent_col = np.where(S == AGENT)
        Agent_row = Agent_row[0]
        Agent_col = Agent_col[0]

        a = 1
        max_Q = -10000
        best_action = []
        # Q学習を使う場合
        for i in range(NUM_ACTION):
            q = Q[Agent_row, Agent_col, ACTION[i]]
            if q > max_Q:
                max_Q = q
                best_action = [ACTION[i]]
            elif q == max_Q:
                best_action.append(ACTION[i])

        # print '>> Best Action,', best_action
        a = np.random.choice(best_action)

        if GREEDY_RATIO < random.random():
            return a
        else:
            return np.random.choice([0, 1, 2, 3])


# 環境クラス
class State(object):
    """ゲームの中身
    S_NEXT, R = goNextStep(S,a)
    """

    def __init__(self):
        self.initS = FIELD

    def __getReward(self, Snext):
        # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'に変化
        Agent_row, Agent_col = np.where(Snext == AGENT)
        row1 = Agent_row[0]
        col1 = Agent_col[0]

        # GOALの場合は,報酬を返却
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
        :param S:
        :param a:
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
            S_next = Snow

        # クラスの外側で、状態判定をする為にOptionを返す
        if row2 == GOAL_ROW and col2 == GOAL_COL:
            option = GOAL
        else:
            option = ROAD

        return S_next, self.__getReward(S_next), option

    def getInitState(self):
        return FIELD
        pass


# 描画用PySideクラス
class GameWindow(QWidget):
    # 定数
    width = NUM_COL
    height = NUM_ROW
    Margin = 20
    interval_time = 0.1

    def __init__(self, parent=None):
        QWidget.__init__(self)

        # クラスプロパティ
        # -----------------
        # 反復数
        self.iter_num = 0

        # ログ用の定数
        self.goaled_number = 0

        # 画面バッファ
        self.pixmap = QPixmap(self.size())

        # 解析用オブジェクト
        self.agent = Agent(numAction=4)
        self.state = State()
        self.field = FIELD
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


        # FIELD[row, col] = AGENT

        # 総当たり学習
        S = self.S
        # S[START_ROW, START_COL] = ROAD
        # S[START_ROW+1, START_COL+1] = AGENT
        # self.iter_cur += 1
        # if self.iter_cur > NUM_ROW*NUM_COL:
        #     self.ter_cur = 0
        # r = int(self.iter_cur / NUM_COL)
        # col =


        o = self.state.encodeStateToO(S)


        # 初期設定
        # S = self.S
        # o = self.state.encodeStateToO(S)

        # 1. エージェントは環境から受け取った観測Sを受け取り、方策planに基づいて環境に行動aを渡す
        a = self.agent.getNextAction(np.copy(S))

        # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'を返却
        S_next, r, option = self.state.getNextState(np.copy(S), a)
        o_next = self.state.encodeStateToO(np.copy(S_next))

        # 3. Agentに学習させる
        self.agent.learn(np.copy(S), a, r, np.copy(S_next))

        # 4. Stateの初期化判定
        if option == GOAL:
            self.goaled_number += 1
            S = self.state.getInitState()
        else:
            S = S_next

        # プロパティの格納
        self.S = np.copy(S)
        self.cursor_pos = np.copy(o_next)

        # 描画準備
        painter = QPainter(self.pixmap)
        self.drawGrid(painter)
        # エージェントの位置をバッファに書き込み
        self.drawState(painter)
        # デバッグ文字をバッファに書き込み
        self.drawDebugLog(painter)
        # 画面更新
        self.update()

    def paintEvent(self, *args, **kwargs):
        # QPainterを生成
        painter = QStylePainter(self)
        # QPainterでバッファに準備したデータを描画
        painter.drawPixmap(0, 0, self.pixmap)

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
        label = 'Goal:%d / Iter:%d' % (self.goaled_number, self.iter_num)
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
        for x in range(NUM_ROW):
            for y in range(NUM_COL):
                if self.field[x][y] == WALL:
                    painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
                elif self.field[x][y] == ROAD:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))

                    _o = [x, y]
                    qval = np.max(self.agent.qlearnobj.getQ()[x][y]) / 1 * 255.
                    # print(self.agent.get_Q_values(_o))

                    if qval != 0:
                        painter.setBrush(QBrush(QColor(qval, 255 - qval, 255 - qval, 255), Qt.SolidPattern))
                        # painter.drawRect(x * dx, y * dy, dx, dy)
                # elif self.field[x][y] == GOAL:
                #     painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                # elif self.field[x][y] == AGENT:
                #     painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
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
        return QSize(self.width * self.Margin, self.height * self.Margin)

    def keyPressEvent(self, event):
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

        else:
            pass

        print 'INTERVAL_TIME', self.interval_time
        self.update()

    pass


def main():
    """
    Q値が伝播しないので、
    配列を渡すときにコピーする。
    解決: learnに渡す配列をコピーした
    """
    agent = Agent()
    state = State()

    # ログ用の定数
    goaled_number = 0

    # 初期設定
    S = state.getInitState()
    print '>> State : Init State'
    print S

    for i in range(MAX_ITERATE):
        if i % (MAX_ITERATE / 20) == 0:
            print i
            # agent.displayQ()



        # 1. エージェントは環境から受け取った観測Sを受け取り、方策planに基づいて環境に行動aを渡す
        a = agent.getNextAction(np.copy(S))
        # print '>>Agent Next Action is :%d' % a

        # 2. 環境StateはエージェントAgentから受け取った行動aと、現在の状態Sにもとづいて、次の状態S'を返却
        S_next, r, option = state.getNextState(np.copy(S), a)
        # print '>>State Next step:'
        # print S_next
        # print '>>> Reward ', r
        # print '>>> Option ', option

        # 3. Agentに学習させる
        agent.learn(np.copy(S), a, r, np.copy(S_next))


        # 4. Stateの初期化判定
        if option == GOAL:
            goaled_number += 1
            S = state.getInitState()
        else:
            S = S_next

    print '>> GOAL NUMBER :', goaled_number
    print agent.displayQ()


def main2():
    app = QApplication(sys.argv)
    win = GameWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main2()
