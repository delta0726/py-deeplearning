# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-8 バッチ正規化
# Module      : P11_model_util.py
# Description : ユーティリティ（モデル定義）
# Created on  : 2021/5/31
# Page        : P112- P113
# ******************************************************************************


from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.utils import plot_model


# 関数定義 : add_conv_pool_layers()
# --- 畳み込み層・プーリング層を追加する関数
def add_conv_pool_layers(x, filters, kernel_size, pool_size, activation='relu'):
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size)(x)
    return x


# 関数定義 : add_dense_layer()
# --- 全結合層を定義する関数
def add_dense_layer(x, dim, use_bn=True, activation='relu'):
    x = Dense(dim, use_bias=not use_bn)(x)
    x = Activation(activation)(x)
    if use_bn:
        x = BatchNormalization()(x)
    return x


def save_model_info(info_file, graph_file, model):
    with open(info_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=graph_file, show_shapes=True)

