# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 2 機械学習とPyTorch
# Theme       :
# Creat Date  : 2022/1/15
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜目次＞
# 1 逆伝播法の連鎖律
# 2 損失関数
# 3 勾配降下法
# 4 PyTorchを使った機械学習


# 0 準備 ----------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1 逆伝播法の連鎖律 ------------------------------------------------------

# ＜ポイント＞
# - PyTorchではbackwardメソッドで逆伝播法を計算することができる


# データ作成
# --- xの勾配を求める際はrequires_gradをTrueにする必要がある
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = (y ** 2).mean()
z.backward()
print(x.grad)


# 2 損失関数 --------------------------------------------------------------

# ＜ポイント＞
# - 回帰問題におけるMSE(平均二乗誤差)の動きを確認する


# データ作成
x_data = [1, 2, 3, 4, 5]
y_data = [3, 6, 9, 12, 15]

# 空リストの定義
w_list = []
MSE_list = []

# 関数定義
# --- x * 傾き
def forward(x):
    return x * w

# 関数定義
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

for w in np.arange(0, 6, 0.5):
    print('w = ', w)
    loss_sum = 0.0

    for x, y in zip(x_data, y_data):
        y_pred = forward(x)
        loss_val = loss(x, y)
        loss_sum += loss_val
        print('\t', x, y, y_pred, loss_val)

    # データ出力
    print('MSE = ', loss_sum / len(x_data))

    # データ格納
    w_list.append(w)
    MSE_list.append(loss_sum / len(x_data))


# プロット作成
plt.plot(w_list, MSE_list)
plt.ylabel('MSE')
plt.xlabel('w')
plt.show()


# 3 勾配降下法 ----------------------------------------------------

# ＜ポイント＞
# - 実際のディープラーニングでは確率的勾配降下法(SGD)が使用される
#   --- ランダムにサンプルを選択してLoss計算を開始することで局所解を回避する


# データ作成
x_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y_data = np.array([3, 6, 9, 12, 15], dtype=np.float32)


# 関数定義
# --- x * 傾き
def forward(x):
    return x * w

# 関数定義
# --- 平均二乗誤差(MSE)
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# 関数定義
# --- 勾配降下法
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()

# パラメータ設定
epochs = 10
lr = 0.01
w = 1

# 空リストの定義
w_list = [w]
loss_list = []

for epoch in range(epochs):
    y_pred = forward(x_data)
    loss_val = loss(y_data, y_pred)
    grad_val = gradient(x_data, y_data, y_pred)
    w = w - lr * grad_val

    # データ格納
    w_list.append(w)
    loss_list.append(loss_val)

    # データ出力
    print(f'epoch {epoch}: w = {w: .3f}: loss {loss_val: .3f}')


# プロット作成
# --- 損失量
plt.plot(loss_list)
plt.show()

# プロット作成
# --- 傾き
plt.plot(w_list)
plt.show()


# 4 PyTorchを使った機械学習 -----------------------------------------

# ＜ポイント＞
# -


# クラス定義
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# 乱数シードの設定
torch.manual_seed(3)

# インスタンス生成
model = Model(1, 1)

# アイテム定義
# --- 損失関数
# --- オプティマイザー
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# データ作成
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float).view(-1, 1)
y = torch.tensor([3, 6, 9, 12, 15], dtype=torch.float).view(-1, 1)

# パラメータ設定
epochs = 1000
loss_list = []
w_list = []

# ループ学習
for epoch in range(epochs):
    y_pred = model.forward(X)
    loss_val = criterion(y_pred, y)

    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()

    loss_list.append(loss_val)
    w_list.append(model.linear.weight.item())


# プロット作成
# --- 損失量
plt.plot(loss_list)
plt.show()

# プロット作成
# --- 傾き
plt.plot(w_list)
plt.show()
