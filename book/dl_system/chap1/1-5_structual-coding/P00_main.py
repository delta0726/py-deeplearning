# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 1 Deep Learningによる画像分類の基礎
# Theme       : 1-5 Sequentialモデルによる実装
# Module      : P00_main.py
# Description : 起動スクリプト
# Created on  : 2021/5/24
# Page        : P23 - P25 / P35 - P36
# ******************************************************************************


# ライブラリ
import os
import sys

# カレントディレクトリの指定
sys.path.append(os.path.join(os.getcwd(), os.path.join("book", "dl_system", "chap1", "1-5")))

from P01_module_maker import ModelMaker


# パラメータ
# --- 定数
DST_DIR = 'D01_estimator'
EST_FILE = os.path.join(DST_DIR, 'estimator.h5')
INFO_FILE = os.path.join(DST_DIR, 'model_info.txt')
GRAPH_FILE = os.path.join(DST_DIR, 'model_praph.pdf')
HIST_FILE = os.path.join(DST_DIR, 'history.pdf')
INPUT_SIZE = (28, 28)
FILTERS = (32, 64)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_DIMS = (1024, 128, 10)
LR = 1e-3
DATA_SIZE = 1000
BATCH_SIZE = 128
EPOCHS = 10
VALID_RATE = 0.2

# インスタンス生成
maker = ModelMaker(dst_dir=DST_DIR,
                   est_file=EST_FILE,
                   info_file=INFO_FILE,
                   graph_file=GRAPH_FILE,
                   hist_file = HIST_FILE,
                   input_size=INPUT_SIZE,
                   filters=FILTERS,
                   kernel_size=KERNEL_SIZE,
                   pool_size=POOL_SIZE,
                   dense_dims=DENSE_DIMS,
                   lr=LR,
                   data_size=DATA_SIZE,
                   batch_size=BATCH_SIZE,
                   epochs=EPOCHS,
                   valid_rate=VALID_RATE)

# 実行
maker.execute()
