# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-6 データ拡張
# Module      : check.py
# Description : 起動スクリプト
# Created on  : 2021/5/29
# Page        : P84 - P91
# ******************************************************************************


# ライブラリ
import os
import sys

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# パス操作
# --- カレントディレクトリの指定
# --- システムパスの追加
cd = os.path.join(os.getcwd(), os.path.join("book", "dl_system", "chap2", "2-6_data-expansion"))
os.chdir(cd)
sys.path.append(cd)


# パラメータ設定
IMG_FILE = 'D00_dataset/training/scissors/scissors_0003.jpg'
PLT_ROW = 1
PLT_COL = 4


# 関数定義 : plot()
# --- 画像処理
def plot(title, img, datagen):
    plt.figure(title)
    i = 0
    for data in datagen.flow(img, batch_size=1):
        plt.subplot(PLT_ROW, PLT_COL, i + 1)
        plt.axis('off')
        plt.imshow(array_to_img(data[0]))
        i += 1
        if i == PLT_ROW * PLT_COL:
            break
        plt.show()


# データ準備
# --- 画像ロード
# --- 画像の配列変換
# --- 次元変換
img = load_img(IMG_FILE, target_size=(160, 100))
img = img_to_array(img)
img = img.reshape((1,) + img.shape)

# 回転
datagen = ImageDataGenerator(rotation_range=30)
plot(title='rotation', img=img, datagen=datagen)

# 水平方向移動
datagen = ImageDataGenerator(width_shift_range=0.2)
plot('width_shift', img, datagen)

# 垂直方向移動
datagen = ImageDataGenerator(height_shift_range=0.2)
plot('height_shift', img, datagen)

# 歪み
datagen = ImageDataGenerator(shear_range=30)
plot('zoom', img, datagen)

# ズーム
datagen = ImageDataGenerator(zoom_range=[0.7, 1.3])
plot('zoom', img, datagen)

# 水平方向の反転
datagen = ImageDataGenerator(horizontal_flip=True)
plot('horizontal_flip', img, datagen)

# 垂直方向の反転
datagen = ImageDataGenerator(vertical_flip=True)
plot('vertical_flip', img, datagen)

# 上記全てのデータ加工
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=30,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True
)

# プロット
plot('all', img, datagen)
