# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 01 箱としての変数
# Created by: Owner
# Created on: 2021/6/24
# Page      : P3 - P7
# ******************************************************************************

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


# 変数定義
data = np.array(1.0)

# クラスに変数を格納＆出力
# --- 以下の操作でクラス内の処理実行が可能となる
# --- self = Variable(data)
x = Variable(data)
print(x.data)
