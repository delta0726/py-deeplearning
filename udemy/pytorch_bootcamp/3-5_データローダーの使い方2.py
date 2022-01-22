# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : データローダーの使い方（ワイン品質データセット）
# Creat Date  : 2022/1/22
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - データローダーは重要性が高いので、ワイン品質データセットで使い方を復習する


# ＜目次＞
# 0 準備
# 1 データセットの確認
# 2 ラベルデータの変更
# 3 モデル構築
# 4 データローダーの定義
# 5 学習
# 6 テストデータの評価


# 0 準備 ----------------------------------------------------------------

# ＜ポイント＞
# - 前回定義したものを活用


# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import math
import collections


# データロード
df = pd.read_csv('data/winequality-white.csv', sep=';')


# 1 データセットの確認 ----------------------------------------------------

# ＜ポイント＞
# - ラベルはマルチクラスの分類問題になっている
#   --- サンプル数の僅少なラベルが存在することを確認


# データ確認
df.head()

# データ情報
df.info()
df.shape

# プロット作成
# --- ラベルのカテゴリ数とサンプル数
# --- カテゴリ数は9つ、サンプル数の少ないカテゴリがある
count_element = df.groupby('quality')['quality'].count()
plt.plot(count_element)
plt.show()


# 2 ラベルデータの変更 ----------------------------------------------------

# ＜ポイント＞
# - 僅少なラベルは予測しにくいので、ラベルカテゴリを３つに再編成する


# ラベルを3カテゴリに変更
y = df['quality'].values
newlabel = []

for val in y:
    if val <= 3:
        newlabel.append(0)
    elif val <= 7:
        newlabel.append(1)
    else:
        newlabel.append(2)

# カテゴリのカウント
# --- 講義の方法
a = collections.Counter(newlabel)
list_a = []
for i in range(len(a)):
    list_a.append(a[i])

# カテゴリのカウント
# --- 簡単な方法
from collections import Counter
Counter(newlabel)

# プロット確認
plt.plot(list_a)
plt.show()


# 3 モデル構築 -------------------------------------------------------------

# ＜ポイント＞
# - 2層のニューラルネットワークモデルを構築


# モデル定義
class Model(nn.Module):
    def __init__(self, in_features=11, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# インスタンス生成
model = Model()

# アイテム設定
# --- 損失関数
# --- オプティマイザー
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 4 データローダーの定義 --------------------------------------------------

# ＜ポイント＞
# - データ取り込みと加工を含めた一連の処理をデータローダーとして定義する


class WineDataset(Dataset):
    def __init__(self):
        # データ取り込み
        df = pd.read_csv('data/winequality-white.csv', sep=';')

        # ラベルを3カテゴリに変更
        y = df['quality'].values
        newlabel = []

        for val in y:
            if val <= 3:
                newlabel.append(0)
            elif val <= 7:
                newlabel.append(1)
            else:
                newlabel.append(2)

        # データ作成
        X = df.drop('quality', axis=1).values
        y = newlabel

        # データ分割
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=123)

        # データ格納
        self.x_data = torch.FloatTensor(X_train)
        self.x_test = torch.FloatTensor(X_test)
        self.y_data = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)

        self.num = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.num


# 5 学習 --------------------------------------------------------------

# ＜ポイント＞
# - データ取得の箇所はデータローダーを用いて行う


# パラメータ設定
epochs = 50
Batch_Size = 500

# 空の配列
loss_list = []

# データローダーの定義
dataset = WineDataset()
trainloader = DataLoader(dataset=dataset, batch_size=Batch_Size, shuffle=True)

# サンプル数の取得
total_samples = len(dataset)
print(total_samples)

# イテレーション回数
n_iterations = math.ceil(total_samples / Batch_Size)
print(n_iterations)

# デバッグ用
# epoch = 1
# temp = trainloader.__iter__()
# inputs, labels = temp.next()

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        print(f'Epoch {epoch+1} / {epochs} Iteration {i+1} / {n_iterations} Inputs {inputs.shape} Label {labels.shape}')

        y_pred = model.forward(inputs.data)
        loss = criterion(y_pred, labels.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss)


# プロット確認
plt.plot(loss_list)
plt.show()


# 6 テストデータの評価 ------------------------------------------------------

# ＜ポイント＞
# - モデル精度の評価はテストデータで行う


# 損失関数から損失量を取得
with torch.no_grad():
    predicted_y = model.forward(dataset.x_test)
    loss = criterion(predicted_y, dataset.y_test)

# 確認
print(loss.item())

# Accuracyの算出
# --- 敢えてループ計算
correct = 0
with torch.no_grad():
    for i, data in enumerate(dataset.x_test):
        y_val = model.forward(data)
        if y_val.argmax().item() == dataset.y_test[i]:
            correct += 1

# 確認
print(f'{correct} out of {len(dataset.y_test)} = {round(100 * correct / len(dataset.y_test), 2)}%')
