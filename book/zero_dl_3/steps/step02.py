# ******************************************************************************
# Title     : ゼロから作るDeep Learning3 （フレームワーク編）
# Stage     : 1 微分を自動で求める
# Chapter   : 02 変数を生み出す関数
# Created by: Owner
# Created on: 2021/6/24
# Page      : P9 - P13
# ******************************************************************************

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


# インスタンス生成
x = Variable(np.array(10))
f = Square()

# 関数実行
y = f(x)

# 確認
print(type(y))
print(y.data)
