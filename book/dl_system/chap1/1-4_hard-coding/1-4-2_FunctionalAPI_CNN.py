# ******************************************************************************
# Title     : Pythonディープラーニングシステム実装法
# Chapter   : 1 Deep Learningによる画像分類の基礎
# Theme     : 1-4-2 FunctionalAPIによる実装
# Created by: Owner
# Created on: 2021/5/24
# Page      : P15 - P17
# ******************************************************************************


# ＜概要＞
# - 複雑な構造のネットワークを表現するにはFunctionalAPIによる実装が適切
#   --- Sequentialモデルは複雑なネットワークを表現するには適切とは言えない
#   ---


# ＜目次＞
# 0 準備
# 1 データ準備
# 2 モデル構築
# 3 モデル訓練


# 0 準備 ---------------------------------------------------------------------------------------

# ライブラリ
from pprint import pprint

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# 1 データ準備 ---------------------------------------------------------------------------------

# データロード
(train_images, train_classes), (_, _) = mnist.load_data()

# データ形状
# --- (60000, 28, 28)
# --- (60000,)
train_images.shape
train_classes.shape

# データ変換
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32') / 255
train_classes = to_categorical(train_classes)

# データ確認
train_images.shape
train_classes.shape

# データ抽出
# --- 訓練時間省略のため1000件のみ抽出
train_images = train_images[:1000]
train_classes = train_classes[:1000]


# 2 モデル構築 ---------------------------------------------------------------------------------

# ＜ポイント＞
# - 入力層と出力層を切り離して設定している
# - 全体的に記述がスッキリしている


# 入力層
input_x = Input(shape=(28, 28, 1))

# 出力層
x = input_x

# Conv1, Conv2, Pool1
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Conv3, Conv4, Pool2
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Flatten
x = Flatten()(x)

# FC1, FC2, Output
x = Dense(1024, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

# モデル全体の入出力を定義
model = Model(input_x, x)


# 3 モデル訓練 ----------------------------------------------------------------

# 訓練条件の設定
model.compile(
    optimizer=Adam(lr=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練
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

# Lossカーブ
pprint(history.history['val_loss'])
