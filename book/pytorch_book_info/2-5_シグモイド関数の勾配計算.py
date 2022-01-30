# ******************************************************************************
# Book        : 最短コースでわかる PyTorch ＆深層学習プログラミング
# Chapter     : 2 Pytorchの基本機能
# Theme       : 5 シグモイド関数の勾配計算
# Creat Date  : 2022/1/30
# Final Update:
# URL         : https://github.com/makaishi2/pytorch_book_info
# Page        : P88 - P93
# ******************************************************************************


# ＜概要＞
# - シグモイド関数の勾配計算も二次関数の場合と同様


# ＜目次＞
# 0 準備
# 1 シグモイド関数の定義
# 2 計算グラフの可視化
# 3 勾配計算
# 4 勾配の初期化


# 0 準備 --------------------------------------------------------------

# ライブラリ
import torch
import matplotlib.pyplot as plt


# データ準備
# --- 等差数列
x_np = np.arange(-2, 2.1, 0.25)

# テンソル変換
x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)

# 確認
print(x)


# 1 シグモイド関数の定義 -------------------------------------------------

# シグモイド関数の定義
sigmoid = torch.nn.Sigmoid()

# yの値の計算
y = sigmoid(x)

# プロット作成
plt.plot(x.data, y.data)
plt.show()


# 2 計算グラフの可視化 ------------------------------------------------

# ＜ポイント＞
# - 勾配計算は数値微分なので、具体的な数値(スカラー)を準備しておく必要がある


# スカラーの計算
# --- 勾配計算のためには最終値はスカラーの必要があるためダミーでsum関数をかける
z = y.sum()

# 可視化関数の呼び出し
g = make_dot(z, params={'x': x})
g


# 3 勾配計算 --------------------------------------------------------

# ＜ポイント＞
# - Pytorchでは勾配計算はbackwardメソッドを適用するだけで完了する


# 勾配計算
z.backward()

# 勾配値の取得
print(x.grad)

# プロット作成
plt.plot(x.data, y.data, c='b', label='y')
plt.plot(x.data, x.grad.data, c='k', label='y.grad')
plt.legend()
plt.show()


# 4 勾配の初期化 --------------------------------------------------------

# 勾配の初期化
x.grad.zero_()
print(x.grad)
