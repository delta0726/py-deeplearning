# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 3 DNN（ディープニューラルネットワーク）
# Theme       : カテゴリデータを含む回帰予測
# Creat Date  : 2022/1/23
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - カテゴリデータを含むデータセットに対して回帰を適用する
#   --- カテゴリカルデータのエンコーディングを行う


# ＜目次＞
# 0 準備
# 1 モデル定義
# 2 データローダーの定義
# 3 モデル構築
# 4 データローダーの動作確認
# 5 学習設定
# 6 学習
# 7 テストデータによる評価
# 8 新しいデータの予測


# 0 準備 ----------------------------------------------------------------

# ＜ポイント＞
# - 前回定義したものを活用


# ライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# データ準備
# --- 中古車の販売価格データ
df = pd.read_csv('data/autos.csv', encoding='ISO-8859-1')

# データ確認
df.info()
df.columns
df.shape

# 使用するデータ
df[["powerPS", "kilometer", "yearOfRegistration", "brand", "vehicleType", "price"]]


# 1 モデル定義 -----------------------------------------------------------

# ＜ポイント＞
# - ドロップアウトやバッチ正規化も実装していく


# モデル定義
class CustomModel(nn.Module):
    def __init__(self, emb_size, n_cont, p=0.5, h1=150, h2=100, h3=30, out_features=1):
        super().__init__()

        # カテゴリカルデータのエンコーディング
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_size])

        # ドロップアウト
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)

        # エンコーディングしたデータの次元数を取得
        n_emb = sum((nf for ni, nf in emb_size))

        # バッチ正規化
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # ネットワーク
        self.fc1 = nn.Linear(n_cont + n_emb, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))

        x = torch.cat(embeddings, 1)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.out(x)
        return x


# 2 データローダーの定義 -------------------------------------------------------

class UsedCarDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('data/autos.csv', encoding='ISO-8859-1')
        df = df[["powerPS", "kilometer", "yearOfRegistration", "brand", "vehicleType", "price"]]
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = df[df['price'] > 100]
        df = df[df['price'] < 30000]
        df = df[df['powerPS'] > 30]

        df['price'] = df['price'] / 100
        df['kilometer'] = df['kilometer'] / 1000
        df['yearOfRegistration'] = df['yearOfRegistration'] / 1000
        df['powerPS'] = df['powerPS'] / 100

        print(df.head())

        categ_cols = ['brand', 'vehicleType']
        contin_cols = ['powerPS','kilometer','yearOfRegistration']
        y_col = ['price']

        for cat in categ_cols:
            df[cat] = df[cat].astype('category')

        br = df['brand'].cat.codes.values
        vt = df['vehicleType'].cat.codes.values
        print(df['brand'].cat.codes.values)
        print(df['brand'].cat.categories)
        print(df['vehicleType'].cat.categories)
        categs = np.stack([br, vt], 1)
        self.categs = torch.tensor(categs, dtype=torch.int64)

        contins = np.stack([df[col].values for col in contin_cols], 1)
        self.contins = torch.tensor(contins, dtype=torch.float)
        
        y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1, 1)

        cat_size = [df[col].cat.categories.size for col in categ_cols]
        self.emb_size = [(size, min(50, (size + 1) // 2)) for size in cat_size]

        data_size = len(y)
        test_size = int(data_size * 0.2)

        self.cat_train = self.categs[:data_size - test_size]
        self.cat_test = self.categs[data_size - test_size:]
        self.con_train = self.contins[:data_size - test_size]
        self.con_test = self.contins[data_size - test_size:]
        self.y_train = y[:data_size - test_size]
        self.y_test = y[data_size - test_size:]

        self.num = len(self.y_train)
    
    def __getitem__(self, index):
        return self.cat_train[index], self.con_train[index], self.y_train[index]

    def __len__(self):
        return self.num


# 4 データローダーの動作確認 --------------------------------------------------------

# インスタンス生成
dataset = UsedCarDataset()

# デバッグ用
# self = dataset

# パラメータ設定
total_sample = len(dataset)

# 確認
print(total_sample)


# 5 学習設定 -----------------------------------------------------------------------

# パラメータ設定
epochs = 15
batch_size = 13000
n_iterations = math.ceil(total_sample / batch_size)

# 空の配列
loss_list = []

# データローダーの定義
trainloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 乱数シードの設定
torch.manual_seed(3)

# モデル構築
model = CustomModel(emb_size=dataset.emb_size, n_cont=3, p=0.3,
                    h1=50, h2=50, h3=30, out_features=1)

# 設定
# --- 損失関数
# --- オプティマイザー
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 6 学習 ------------------------------------------------------------------------

# デバッグ用
# epoch = 0
# i = 0
# temp = trainloader.__iter__()
# cat_train, con_train, labels = temp.next()

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        cat_train, con_train, labels = data

        # 予測値の取得
        y_pred = model(cat_train, con_train)

        # 損失関数の値を取得
        loss = torch.sqrt(criterion(y_pred, labels))
        loss_list.append(loss)

        # 学習
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 進捗表示
        print(f'Epoch {epoch + 1} / {epochs} '
              f'Iteration {i + 1} / {n_iterations}, '
              f'Category {cat_train.shape} '
              f'Continuous {con_train.shape} '
              f'Loss {loss.item():3f}')


# プロット作成
# --- 損失量の推移
plt.plot(loss_list)
plt.show()

# モデル保存
torch.save(model.state_dict(), 'model/GermanyCarModel.pt')


# 7 テストデータによる評価 -------------------------------------------------------

# RMSEの計算
with torch.no_grad():
    y_val = model(dataset.cat_test, dataset.con_test)
    loss = torch.sqrt(criterion(y_val, dataset.y_test))
    print(f'{loss.item():.3f}')

print(f'Predicted_y   True_y   Error')
for i in range(30):
    error = np.abs(y_val[i].item() - dataset.y_test[i].item())
    print(f'{i + 1}.'
          f'{y_val[i].item(): .3f}   '
          f'{dataset.y_test[i].item(): .3f},   '
          f'{error: .3f}')


# 8 新しいデータの予測 --------------------------------------------------------

# モード切替
model.eval()

# データ作成
xcats = [[1, 3]]
xcats = torch.tensor(xcats, dtype=torch.int64)
xcont = [[1.9, 236, 0.2, 2.0111]]

# 予測
with torch.no_grad():
    predict = model(xcats,xconts)
    print(predict)
