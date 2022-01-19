# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : ロジスティック回帰
# Creat Date  : 2022/1/19
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - Pytorchを使ってロジスティック回帰を行う
#   --- 損失関数やオプティマイザーの定義など、独自のプロセスが含まれる


# ＜目次＞
# 0 準備
# 1 データセットの確認
# 2 モデルデータの作成
# 3 モデル構築
# 4 シミュレーション
# 5 モデル精度の評価


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# データロード
data = datasets.load_breast_cancer()


# 1 データセットの確認 ------------------------------------------------------

# 確認用データ
# --- Pandas DataFrame
df_X = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=['target'])

# データ確認
df_X
df_X.columns
df_X.shape
df_y


# 2 モデルデータの作成 ------------------------------------------------------

# モデル用データ
# --- Numpy配列
X, y = data.data, data.target

# パラメータ取得
# --- サンプル数と特徴量数
n_samples, n_features = X.shape

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# データ基準化
# --- 訓練データの平均値と標準偏差に基づいてテストデータを基準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# データ変換
# --- テンソル型に変換
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)


# 3 モデル構築 ----------------------------------------------------------------

# モデル定義
class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# インスタンス生成
model = Model(n_features)

# アイテム設定
# --- 損失関数
# --- オプティマイザー
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 4 シミュレーション ------------------------------------------------------------

# パラメータ設定
epochs = 1000

# 空の配列
loss_list = []


for epoch in range(epochs):

    # 設定
    # --- モデル定義
    # --- 損失関数の設定
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # 学習
    # --- オプティマイザーの勾配の初期化
    # --- 勾配の計算
    # --- 学習
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 損失量の格納
    loss_list.append(loss.item())

    # 表示
    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item(): 4f}')


# プロット表示
# --- 損失量
plt.plot(loss_list)
plt.show()


# 5 モデル精度の評価 -------------------------------------------------------

# モデル精度の評価
# --- 予測値の取得
# --- Accuracyの算出
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_0_1 = y_predicted.round()
    acc = y_predicted_0_1.eq(y_test).sum() / float(y_test.shape[0])
    print(round(acc.item(), 3))
