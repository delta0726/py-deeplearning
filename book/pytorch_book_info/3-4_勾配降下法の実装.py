# ******************************************************************************
# Book        : 最短コースでわかる PyTorch ＆深層学習プログラミング
# Chapter     : 3 初めての機械学習
# Theme       : 4 勾配降下法の実装
# Creat Date  : 2022/1/31
# Final Update:
# URL         : https://github.com/makaishi2/pytorch_book_info
# Page        : P105 - P107
# ******************************************************************************


# ＜概要＞
# - 書籍(3-3)で説明された｢山登りアルゴリズム｣をプログラミングで実装する
#   --- 現在地の勾配を計算して勾配が急な方角を目指して頂上を目指す
#   --- 以下ではP107の図を作成


# ＜目次＞
# 0 準備
# 1 関数定義
# 2 データ作成
# 3 勾配降下法のシミュレーション
# 4 プロット作成


# 0 準備 ------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import numpy as np


# 1 関数定義 ---------------------------------------------------------------

def L(u, v):
    return 3 * u ** 2 + 3 * v ** 2 - u * v + 7 * u - 7 * v + 10


def Lu(u, v):
    return 6 * u - v + 7


def Lv(u, v):
    return 6 * v - u - 7


# 2 データ作成 ------------------------------------------------------------

# データ作成
u = np.linspace(-5, 5, 501)
v = np.linspace(-5, 5, 501)

# 格子座標の作成
U, V = np.meshgrid(u, v)
Z = L(U, V)


# 3 勾配降下法のシミュレーション -------------------------------------------

# 勾配降下法のシミュレーション
W = np.array([4.0, 4.0])
W1 = [W[0]]
W2 = [W[1]]
N = 21
alpha = 0.05
for i in range(N):
    W = W - alpha * np.array([Lu(W[0], W[1]), Lv(W[0], W[1])])
    W1.append(W[0])
    W2.append(W[1])

n_loop = 11

WW1 = np.array(W1[:n_loop])
WW2 = np.array(W2[:n_loop])
ZZ = L(WW1, WW2)


# 4 プロット作成 -----------------------------------------------------------

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_zlim(0, 250)
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('loss')
ax.view_init(50, 240)
ax.xaxis._axinfo["grid"]['linewidth'] = 2.
ax.yaxis._axinfo["grid"]['linewidth'] = 2.
ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.contour3D(U, V, Z, 100, cmap='Blues', alpha=0.7)
ax.plot3D(WW1, WW2, ZZ, 'o-', c='k', alpha=1, markersize=7)
plt.show()
