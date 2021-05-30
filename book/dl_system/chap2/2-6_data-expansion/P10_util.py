# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-6 データ拡張
# Module      : P10_util.py
# Description : ユーティリティ（全般）
# Created on  : 2021/5/29
# Page        : P91 - P95
# ******************************************************************************


import os
import shutil

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 関数定義 : mkdir()
# --- 指定された名称のディレクトリを作成する関数
def mkdir(d, rm=False):
    if rm:
        # 既存の同名ディレクトリがあれば削除
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    else:
        # 既存の同名ディレクトリがある場合は何もしない
        try:
            os.makedirs(d)
        except FileExistsError:
            pass


# 関数定義 : make_generator()
# --- 訓練データセットを読込むジェネレータを作成する関数
def make_generator(src_dir, valid_rate, input_size, batch_size):

    # インスタンス作成
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=30,
        zoom_range=[0.7, 0.3],
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=valid_rate
    )

    # ジェネレータ作成
    # --- 訓練用データ
    train_generator = train_datagen.flow_from_directory(
        directory=src_dir,
        target_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='training'
    )

    # ジェネレータ作成
    # --- 検証用データ
    valid_generator = train_datagen.flow_from_directory(
        directory=src_dir,
        target_size=input_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical',
        subset='validation'
    )

    # ラッピング
    # --- 訓練データジェネレータ
    trans_ds = Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [None, *train_generator.image_shape],
            [None, train_generator.num_classes]
        )
    )

    # ラッピング
    # --- 検証データジェネレータ
    valid_ds = Dataset.from_generator(
        lambda: valid_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [None, *valid_generator.image_shape],
            [None, valid_generator.num_classes]
        )
    )

    # 各Datasetを無限に繰り返す設定
    trans_ds = trans_ds.repeat()
    trans_ds = trans_ds.repeat()

    return trans_ds, train_generator.n, valid_ds, valid_generator.n


# 関数定義 : plot()
# --- 訓練状況を可視化する関数
def plot(history, filename):
    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='validation')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    # 描画サイズを指定
    plt.figure(figsize=(10, 10))

    # x軸のデータ取得（エポック数）
    xdata = range(1, 1 + len(history['loss']))

    # プロット作成
    # --- 損失
    # --- 正解率
    add_subplot(2, 1, 1, xdata, history['loss'], history['val_loss'], (0, 5), 'loss')
    add_subplot(2, 1, 1, xdata, history['loss'], history['accuracy'], (0, 5), 'loss')

    # 保存
    plt.savefig(filename)
    plt.close('all')
