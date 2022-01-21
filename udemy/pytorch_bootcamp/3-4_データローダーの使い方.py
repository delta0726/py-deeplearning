# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : データローダーの使い方
# Creat Date  : 2022/1/22
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜ポイント＞
# - データローダーを用いてバッチ学習を行う


# ＜目次＞
# 0 準備
# 1 モデル定義
# 2 モデル構築
# 3 データクラスの定義
# 4 データローダーの定義
# 5 シミュレーション


# 0 準備 ----------------------------------------------------------------

# ＜ポイント＞
# - 前回定義したものを活用


# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader   # 追加
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import math


# 1 モデル定義 ---------------------------------------------------------

# ＜ポイント＞
# - 前回定義したものを活用


# モデル定義
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# 2 モデル構築 ---------------------------------------------------------

# ＜ポイント＞
# - 前回定義したものを活用


# 乱数シードの設定
torch.manual_seed(3)

# インスタンス生成
model = Model()

# アイテム設定
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 3 データクラスの定義 ------------------------------------------------

# ＜ポイント＞
# - データローダーで使用するデータクラスを定義する
#   --- __init__    ：データ取得と前処理を行う
#   --- __getitem__ ：出力用データの定義
#   --- __len__     ：データサイズの定義


class IrisDataset(Dataset):
    def __init__(self):
        # データ準備
        df = pd.read_csv('data/iris.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values

        # データ分割
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=123)

        # データ変換
        self.x_data = torch.FloatTensor(X_train)
        self.x_test = torch.FloatTensor(X_test)
        self.y_data = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)

        # プロパティ設定
        # --- データの長さ
        self.datalen = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.datalen


# 4 データローダーの定義 ------------------------------------------------

# パラメータ設定
# --- バッチサイズ
# --- エポック数
Batch_Size = 20
epochs = 100

# 空の配列
loss_list = []

# インスタンス生成
dataset = IrisDataset()

# データローダーの定義
trainloader = DataLoader(dataset=dataset, batch_size=Batch_Size, shuffle=True)

# データ数の確認
# --- データ数は150 * (1 - 0.2) = 120
total_samples = len(dataset)
print(total_samples)

# イテレーション回数の確認
n_iterations = math.ceil(total_samples / Batch_Size)
print(n_iterations)


# 5 シミュレーション --------------------------------------------------

# ＜ポイント＞
# - エポックごとにデータを取り出して学習する

# ＜参考＞
# PytorchでDataLoaderからデータを取り出す
# https://tzmi.hatenablog.com/entry/2020/03/01/202349


# デバッグ用
# epoch = 1
# temp = trainloader.__iter__()
# inputs, labels = temp.next()

for epoch in range(epochs):
    for i, data in enumerate(trainloader):

        # データ取得
        inputs, labels = data

        # 進捗表示
        print(f'Epoch:{epoch + 1} / {epochs}, Iteration {i + 1} / {n_iterations}, Inputs {inputs.shape}, Labels {labels.shape}')

        # 設定
        # --- モデル定義
        # --- 損失関数の設定
        y_pred = model.forward(inputs.data)
        loss = criterion(y_pred, labels.data)

        # 学習
        # --- オプティマイザーの勾配の初期化
        # --- 勾配の計算
        # --- 学習
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失量の格納
        loss_list.append(loss)


# プロット作成
plt.plot(loss_list)
plt.show()