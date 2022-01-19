# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : 多層ニューラルネットワーク
# Creat Date  : 2022/1/19
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - Pytorchを使ってirisのマルチクラス分類のディープラーニングを体験する


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 モデル定義
# 3 モデル構築
# 4 シミュレーション
# 5 予測精度の確認
# 6 モデルのモード切替
# 7 モデル保存


# 0 準備 ----------------------------------------------------------------

# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt


# データロード
df = pd.read_csv('data/iris.csv')

# データ確認
df.head()


# 1 データ準備 -----------------------------------------------------------

# 配列作成
X = df.drop('target', axis=1).values
y = df['target'].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# テンソルに変換
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# 2 モデル定義 -----------------------------------------------------------

# ＜ポイント＞
# - マルチクラス分類では出力層でソフトマックス関数を使う
#   --- Modelには実装不要（CrossEntropyLossの内部で計算）


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


# 3 モデル構築 -----------------------------------------------------------

# ＜ポイント＞
# - マルチクラス分類では出力層でソフトマックス関数を使う
#   --- CrossEntropyLossは内部でソフトマックス関数の計算をしている（Modelには実装不要）


# 乱数シードの設定
torch.manual_seed(3)

# インスタンス生成
model = Model()

# アイテム設定
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 4 シミュレーション ---------------------------------------------------------

# パラメータ設定
epochs = 100

# 空の配列
loss_list = []

epoch = 1
for epoch in range(epochs):
    # 設定
    # --- モデル定義
    # --- 損失関数の設定
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    # 学習
    # --- オプティマイザーの勾配の初期化
    # --- 勾配の計算
    # --- 学習
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 損失量の格納
    loss_list.append(loss)

    # 表示
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}  loss: {loss.item(): .4f}')


# プロット表示
plt.plot(loss_list)
plt.show()


# 5 予測精度の確認 ---------------------------------------------------------

# ＜ポイント＞
# - モデル評価はアウトサンプルのテストデータで行う


# 予測精度の確認
# --- テストデータ
with torch.no_grad():
    predicted_y = model.forward(X_test)
    loss = criterion(predicted_y, y_test)

# 確認
print(loss.item())


# 6 モデルのモード切替 -----------------------------------------------------

# モデルを評価モードに切り替え
model.eval()

# 確認
print(model)

# 未知のデータを予測
# --- 特徴量だけをインプット
new_iris = torch.tensor([5.6, 3.7, 2.1, 0.7])

# 予測
with torch.no_grad():
    print(model(new_iris))
    print(model(new_iris).argmax())


# 7 モデル保存 -------------------------------------------------------------

# ＜ポイント＞
# - モデルはオブジェクトとして保存することができる


# 保存
torch.save(model.state_dict(), 'model/IrisClassificationModel.pt')

# ロード
# --- インスタンス生成
# --- パラメータのロード
# --- 評価モードに切り替え
new_model = Model()
new_model.load_state_dict(torch.load('model/IrisClassificationModel.pt'))
new_model.eval()

# 未知のデータを予測
# --- 特徴量だけをインプット
new_iris = torch.tensor([5.6, 3.7, 2.1, 0.7])

# 予測
with torch.no_grad():
    print(model(new_iris))
    print(model(new_iris).argmax())
    print(new_model(new_iris))
    print(new_model(new_iris).argmax())
