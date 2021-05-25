# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 1 Deep Learningによる画像分類の基礎
# Theme       : 1-5 Sequentialモデルによる実装
# Module      : P10_util.py
# Description : 一般的な処理をまとめたユーティリティプログラム
# Created on  : 2021/5/24
# Page        : P20 - P21
# ******************************************************************************

import os
import shutil
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# 関数定義 : mkdir()
# --- 指定された名称のディレクトリを作成する関数
def mkdir(d, rm=False):
    if rm:
        shutil.rmtree(d, ignore_errors=True)
        os.mkdir(d)
    else:
        try:
            os.mkdir(d)
        except FileExistsError:
            pass


# 関数定義 : load_data()
# --- 訓練用データセットを取得する関数
def load_data(data_size=-1):
    # データロード
    (train_images, train_classes), (_, _) = mnist.load_data()

    # データ加工
    train_images = train_images.reshape(60000, 28, 28, 1)
    train_images = train_images.astype('float32') / 255
    train_classes = to_categorical(train_classes)

    if data_size > len(train_images):
        print('[ERROR] data_size should be less than or equal to len(train_images).')
        sys.exit(0)
    elif data_size == -1:
        # データ数の指定がない場合はデータセットサイズを取得するデータ数とする
        data_size = len(train_images)

    return train_images[:data_size], train_classes[:data_size]


def plot(history, filename):
    # 訓練状況の折れ線グラフを描画する関数
    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='validation')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0,1), loc='lower left')

    # 描画サイズを指定
    plt.figure(figsize=(10, 10))

    # エポック数を取得
    xdata = range(1, 1 + len(history['loss']))

    # 損失の可視化
    # --- 検証用データ
    add_subplot(2, 1, 1, xdata, history['loss'], history['val_loss'], (0, 5), 'loss')
    add_subplot(2, 1, 2, xdata, history['accuracy'], history['val_accuracy'], (0, 1), 'accuracy')

    # ファイル保存
    plt.savefig(filename)
    plt.close('all')
