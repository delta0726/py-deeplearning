# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-4 画像分類ニューラルネットワークの実装
# Module      : P01_module_maker.py
# Description : 起動スクリプト
# Created on  : 2021/5/29
# Page        : P74 - P79
# ******************************************************************************


from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil


class ModelMaker:
    def __init__(self, src_dir, dst_dir, est_file, info_file, graph_file, hist_file,
                 input_size, filters, kernel_size, pool_size, dense_disms, lr,
                 batch_size, epochs, valid_rate):

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.est_file = est_file
        self.info_file = info_file
        self.graph_file = graph_file
        self.hist_file = hist_file
        self.input_size = input_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_disms = dense_disms
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_rate = valid_rate

    def define_model(self):
        # 入力層の定義
        input_x = Input(shape=(*self.input_size, 3))
        x = input_x

        # 畳み込み層・プーリング層の定義
        for f in self.filters:
            x = mutil.add_conv_pool_layers(x, f, self.kernel_size, self.pool_size)

        # 平滑化層の定義
        x = Flatten()(x)

        # 全結合層の定義
        for dim in self.dense_disms[:-1]:
            x = mutil.add_dense_layer(x, dim)

        # 出力層の定義
        x = mutil.add_dense_layer(x, self.dense_disms[-1], activation='softmax')

        # モデル全体の入出力の定義
        model = Model(input_x, x)

        # モデル訓練状況の整理
        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def fit_model(self):
        # データセットを読込むためのジェネレータ、データセットサイズを取得
        train_generator, train_n, valid_generator, valid_n = util.make_generator(
            self.src_dir, self.valid_rate, self.input_size, self.batch_size
        )

        # モデル定義
        model = self.define_model()

        # 訓練の実行
        history = model.fit(
            train_generator,
            steps_per_epoch=int(train_n / self.batch_size),
            epochs=self.epochs,
            validation_data=valid_generator,
            validation_steps=int(valid_n / self.batch_size)
        )

        return model, history.history

    def execute(self):
        # モデルを訓練
        model, history = self.fit_model()

        # ディレクトリ作成
        util.mkdir(d=self.dst_dir, rm=True)

        # モデル保存
        # --- 容量が大きいのでモデル保存は中止
        # model.save(self.est_file)

        # ネットワーク構造の保存
        mutil.save_model_info(info_file=self.info_file, graph_file=self.graph_file, model=model)

        # 訓練状況の保存
        util.plot(history=history, filename=self.hist_file)

        # 損失量の表示
        # --- 最終エポックの検証用データにおける損失
        print('val_loss: %f' % history['val_loss'][-1])
