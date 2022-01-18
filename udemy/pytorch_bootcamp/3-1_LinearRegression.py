# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : 線形回帰
# Creat Date  : 2022/1/19
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - Pytorchを使って線形回帰を行う
#   --- 損失関数やオプティマイザーの定義など、独自のプロセスが含まれる


# ＜目次＞
# 0 準備
# 1 テンソル型のデータ作成
# 2 シミュレーション設定
# 3 シミュレーション
# 4 予測値の計算


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# データ準備
X_gen,  y_gen = datasets.make_regression(n_samples=100, n_features=1,
                                         noise=10, random_state=2)

# データ確認
plt.scatter(x=X_gen, y=y_gen)
plt.show()


# 1 テンソル型のデータ作成 -------------------------------------------------

# データ変換
# --- Numpy配列からtorchのテンソル型に変換
X = torch.from_numpy(X_gen.astype(np.float32))
y = torch.from_numpy(y_gen.astype(np.float32)).view(-1, 1)

# パラメータ取得
# --- サンプル数と特徴量数
n_samples, n_features = X.shape


# 2 シミュレーション設定 ---------------------------------------------------

# インスタンス生成
# --- 線形回帰モデル
model = nn.Linear(1, 1)

# アイテム定義
# --- 損失関数
# --- オプティマイザー
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# パラメータ設定
epochs = 800

# 空の配列
loss_list = []


# 3 シミュレーション -----------------------------------------------------

#epoch = 1
for epoch in range(epochs):

    # モデル定義
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 学習
    # --- オプティマイザーの勾配の初期化
    # --- 勾配の計算
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 損失量の出力
    loss_list.append(loss.item())

    # 表示出力
    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item(): .4f}')


# 結果確認
# --- 損失関数のプロット
plt.plot(loss_list)
plt.show()


# 4 予測値の計算 -----------------------------------------------------

# 予測値の計算
with torch.no_grad():
    predicted_y = model(X).detach().numpy()

    plt.plot(X_gen, y_gen, 'ro')
    plt.plot(X_gen, predicted_y, 'b')
    plt.show()
