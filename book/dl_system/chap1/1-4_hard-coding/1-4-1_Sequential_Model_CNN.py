# ******************************************************************************
# Title     : Pythonディープラーニングシステム実装法
# Chapter   : 1 Deep Learningによる画像分類の基礎
# Theme     : 1-4-1 Sequentialモデルによる実装
# Created by: Owner
# Created on: 2021/5/12
# Page      : P9 - P14
# ******************************************************************************


# ＜概要＞
# - Sequentialモデルは層を線形にスタックしたシンプルなネットワークを実装する
#   --- Sequentialクラスのインスタンスを作成して、addメソッドで層を追加していく


# ＜目次＞
# 0 準備
# 1 データ取り込み
# 2 モデル構築
# 3 コンパイル
# 4 モデル訓練


# 0 準備 ---------------------------------------------------------------------------------------

# ライブラリ
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# 1 データ取り込み -------------------------------------------------------------------------------

# データロード
# --- 今回は訓練データしか使用しない
# --- 検証データはアンダースコアで受け取って変数名は与えない
(train_images, train_classes), (_, _) = mnist.load_data()

# データ形状
# --- (60000, 28, 28)
# --- (60000,)
train_images.shape
train_classes.shape

# データ加工
# --- チャンネルに1を指定（モノクロ画像を意味する）
# --- 特徴量データの0-1にスケーリング
# --- ラベルをOne-Hotでカテゴリカルデータに変換（元データは数値）
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32') / 255
train_classes = to_categorical(train_classes)

# データ形状
# --- (60000, 28, 28, 1)
train_images.shape

# データ抽出
# --- 訓練時間省略のため1000件のみ抽出
train_images = train_images[:1000]
train_classes = train_classes[:1000]


# 2 モデル構築 ---------------------------------------------------------------------------------

# ＜ポイント＞
# - Sequentialもで腕は、入力層と出力層を切り分けていない
#   --- 入力層はConv-1を構築する際に設定されている　


# インスタンス生成
# --- ベースモデル
model = Sequential()

# モデル構築
# --- Conv-1
# --- Conv-2
# --- Pool-1
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3 コンパイル ---------------------------------------------------------------------------

model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 4 モデル訓練 ----------------------------------------------------------------------------

history = model.fit(
    x = train_images,
    y = train_classes,
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

# 結果確認
# --- 最終誤差
print('val_loss: %f' % history.history['val_loss'][-1])
