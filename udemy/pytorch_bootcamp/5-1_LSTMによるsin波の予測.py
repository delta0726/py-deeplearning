# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 5 RNN（リカレントニューラルネットワーク）
# Theme       : LSTMによるsin波の予測
# Creat Date  : 2022/1/24
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - sin波のダミーデータをLSTMで予測する
#   --- LSTMのデータの扱い方とフローを学ぶ


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 時系列ごとのラベル作成
# 3 モデル構築
# 4 学習
# 5 検証データにおける予測
# 6 未知のデータにおける予測


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1 データ作成 ------------------------------------------------------------

# データ準備
x = torch.linspace(0, 499, steps=500)
y = torch.sin(x * 2 * 3.1416 / 30) + 0.05 * torch.randn(500)

# プロット作成
# --- データ確認
plt.figure(figsize=(12, 5))
plt.grid(True)
plt.plot(x, y)
plt.show()

# データ分割
test_size = 30
train_original_data = y[:-test_size]
test_original_data = y[-test_size:]


# 2 時系列ごとのラベル作成 ---------------------------------------------------

# ＜ポイント＞
# - 最初のN個でN+1番目をラベルとするデータ生成器を作成する
#   --- 時系列のトレーニングに使用


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
window_size = 30
train_data = sequence_creator(train_original_data, window_size)


# 3 モデル構築 --------------------------------------------------------------

# モデル定義
class LSTM_Model(nn.Module):
    def __init__(self, in_size=1, h_size=50, out_size=1):
        super().__init__()

        self.h_size = h_size
        self.lstm = nn.LSTM(in_size, h_size)
        self.fc = nn.Linear(h_size, out_size)

        self.hidden = (torch.zeros(1, 1, h_size), torch.zeros(1, 1, h_size))

    def forward(self, sequence_data):
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data), 1, 1), self.hidden)
        pred = self.fc(lstm_out.view(len(sequence_data), -1))

        return pred[-1]


# シード設定
torch.manual_seed(3)

# インスタンス生成
model = LSTM_Model()

# 設定
# --- 損失関数（回帰問題なのでMSE）
# --- オプティマイザー
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 4 学習 ---------------------------------------------------------------------

# パラメータ設定
epochs = 10
test = 30

# 空の配列
loss_list = []

for epoch in range(epochs):
    for sequence_in, y_train in train_data:
        y_pred = model(sequence_in)
        loss = criterion(y_pred, y_train)

        # 初期化
        # --- オプティマイザー
        # --- 隠れ層
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))

        loss.backward()
        optimizer.step()

    loss_list.append(loss)
    print(f'Epoch {epoch+1}  Loss {loss.item():.3f}')


# プロット作成
plt.plot(loss_list)
plt.show()


# 5 検証データにおける予測 ------------------------------------------------

# 検証データの取得
predict_list = []
predict_list = train_original_data[-window_size:].tolist()

# モード切替
model.eval()

# シミュレーション
for i in range(test):
    sequence_in = torch.FloatTensor(predict_list[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.h_size),
                        torch.zeros(1, 1, model.h_size))
        predict_list.append(model(sequence_in).item())

loss = criterion(torch.tensor(predict_list[-window_size:]), y[470:])
print(f'Loss {loss.item():.3f}')

# プロット作成
plt.figure(figsize=(12, 5))
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(470, 500), predict_list[-window_size:])
plt.show()


# 6 未知のデータにおける予測 -----------------------------------------------

# パラメータ設定
epochs = 5
window_size = 30
unknown = 30

# モード変換
# --- 訓練モード
model.train()

# データ定義
full_data = sequence_creator(y, window_size)

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

    print(f'epoch {epoch + 1} Loss {loss.item():.3f}')


# モード切替
# --- 評価モード
model.eval()

# 予測データの定義
predict_list = y[-window_size:].tolist()

# 予測
for i in range(unknown):
    sequence = torch.FloatTensor(predict_list[-window_size:])

    with torch.no_grad():
        model.hidden = (torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))
        predict_list.append(model(sequence).item())

# プロット作成
plt.figure(figsize=(12, 5))
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(470, 500), predict_list[-window_size:])
plt.show()
