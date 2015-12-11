# -*- coding: utf-8 -*-
"""
ndprint.py
ndarrayをプリントする関数

"""

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



if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    for x, y in zip(X, y):
        print 'X:%s, y:%0.2f' % (ndprint(x), y)