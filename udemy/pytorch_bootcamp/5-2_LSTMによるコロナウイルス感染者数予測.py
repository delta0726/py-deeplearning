# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 5 RNN（リカレントニューラルネットワーク）
# Theme       : LSTMによるコロナウイルス感染者数予測
# Creat Date  : 2022/1/26
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - コロナウイルス感染者数予測で時系列データを作成してLSTMによる予測を行う


# ＜目次＞
# 0 準備
# 1 モデル用データの作成
# 2 データ分割
# 3 時系列ごとのラベル作成
# 4 モデル構築
# 5 学習
# 6 テストデータによる検証
# 7 将来データの予測


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler


# データ準備
# --- コロナウイルス感染者数データ
df = pd.read_csv('data/time_series_covid19_confirmed_global.csv')


# 1 モデル用データの作成 -----------------------------------------------------

# ＜ポイント＞
# - 国ごとのデイリーデータを日次で合計してグローバルのデータとする
#   --- 講義と同じデータ期間を抽出したが、値は異なっているようだ


# データ確認
df.head()
df.columns
df.info()

# データ抽出
df = df.iloc[:, 37:83]
df.head()

# データ集計
# --- グローバルの値を出すために列ごとに合計する
daily_global = df.sum(axis=0)
daily_global.index = pd.to_datetime(daily_global.index)

# データ確認
daily_global

# プロット作成
plt.plot(daily_global)
plt.show()


# 2 データ分割 -------------------------------------------------------------

# 数値型に変換
y = daily_global.values.astype(float)

# データ分割
test_size = 3
train_original_data = y[:-test_size]
test_original_data = y[-test_size:]

# データ正規化
scaler = MinMaxScaler(feature_range=(-1, 1))
train_normalized = scaler.fit_transform(train_original_data.reshape(-1, 1))
train_normalized.shape

# テンソル型に変換
train_normalized = torch.FloatTensor(train_normalized).view(-1)


# 3 時系列ごとのラベル作成 ---------------------------------------------------

# ＜ポイント＞
# - 最初のN個でN+1番目をラベルとするデータ生成器を作成する
#   --- 5-1で作成したものを流用


# 関数定義
def sequence_creator(input_data, window):
    dataset = []
    data_len = len(input_data)
    for i in range(data_len - window):
        window_fr = input_data[i:i + window]
        label = input_data[i + window:i + window + 1]
        dataset.append((window_fr, label))
    return dataset

# 動作確認
window_size = 3
train_data = sequence_creator(train_normalized, window_size)


# 4 モデル構築 -------------------------------------------------------------

# モデル定義
class LSTM_Corona(nn.Module):
    def __init__(self, in_size=1, h_size=30, out_size=1):
        super().__init__()
        self.h_size = h_size
        self.lstm = nn.LSTM(in_size, h_size)
        self.fc = nn.Linear(h_size, out_size)

        self.hidden = (torch.zeros(1, 1, h_size),
                       torch.zeros(1, 1, h_size))

    def forward(self, sequence_data):
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data), 1, -1), self.hidden)
        pred = self.fc(lstm_out.view(len(sequence_data), -1))

        return pred[-1]


# シード設定
torch.manual_seed(3)

# インスタンス生成
model = LSTM_Corona()

# 設定
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5 学習 -----------------------------------------------------------------

# パラメータ設定
epochs = 100

# シミュレーション
for epoch in range(epochs):
    for sequence_in, y_train in train_data:
        y_pred = model(sequence_in)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} Loss {loss.item():.3f}')


# 6 テストデータによる検証 ---------------------------------------------------

# パラメータ設定
test = 3

# データ抽出
# --- テストデータ
preds = train_normalized[-window_size:].tolist()

# モード切替
model.eval()

# 予測
for i in range(test):
    sequence = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))
        preds.append(model(sequence).item())

# データ再変換
# --- 元のスケールに戻す
predictions = scaler.inverse_transform(np.array(preds[window_size:]).reshape(-1, 1))

# データ比較
daily_global[-3:]
predictions

# 日付ラベル
x = np.arange('2020-04-07', '2020-04-10', dtype='datetime64[D]').astype('datetime64[D]')
x

# プロット作成
plt.figure(figsize=(12,5))
plt.grid(True)
plt.plot(daily_global)
plt.plot(x,predictions)
plt.show()


# 7 将来データの予測 ------------------------------------------------------------

# モード変換
model.train()

# パラメータ設定
epochs = 200

# データ作成
y_normalized = scaler.fit_transform(y.reshape(-1, 1))
y_normalized = torch.FloatTensor(y_normalized).view(-1)
full_data = sequence_creator(y_normalized, window_size)

# 再学習
for epoch in range(epochs):
    for sequence_in, y_train in full_data:
        y_pred = model(sequence_in)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} Loss {loss.item():.3f}')


# 将来データの抽出
future = 3
preds = y_normalized[-window_size:].tolist()

# モード変換
model.eval()

# 再学習
for i in range(future):
    sequence = torch.FloatTensor(preds[-window_size:])

    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))
        preds.append(model(sequence).item())

predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# 将来日付の作成
x = np.arange('2020-04-10','2020-04-13', dtype='datetime64[D]').astype('datetime64[D]')

# プロット作成
plt.figure(figsize=(12,5))
plt.title('The number of person affected by Corona virus globally')
plt.grid(True)
plt.plot(daily_global)
plt.plot(x, predictions[window_size:])
plt.show()
