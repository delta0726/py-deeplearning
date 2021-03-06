# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-4 画像分類ニューラルネットワークの実装
# Module      : P11_model_util.py
# Description : ユーティリティ（モデル定義）
# Created on  : 2021/5/26
# Page        : P71 - P72
# ******************************************************************************


from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.utils import plot_model


# 関数定義：add_conv_pool_layers()
# --- 畳み込み層とプーリング層を追加する関数
def add_conv_pool_layers(x, filters, kernel_size, pool_size, activation='relu'):
    x = Conv2D(filters, kernel_size=kernel_size, padding='same', activation=activation)(x)
    x = Conv2D(filters, kernel_size=kernel_size, padding='same', activation=activation)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    return x


# 関数定義 : add_dense_layer()
# --- 全結合層を追加する関数
def add_dense_layer(x, dim, activation='relu'):
    x = Dense(dim, activation=activation)(x)
    return x


# 関数定義 : save_model_info()
# --- ネットワーク構造を可視化して保存する関数
def save_model_info(info_file, graph_file, model):
    with open(info_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=graph_file, show_shapes=True)
