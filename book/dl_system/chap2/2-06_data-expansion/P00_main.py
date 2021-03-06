# ******************************************************************************
# Title       : Pythonディープラーニングシステム実装法
# Chapter     : 2 Deep Learningによる画像分類の応用
# Theme       : 2-6 データ拡張
# Module      : P00_main.py
# Description : 起動スクリプト
# Created on  : 2021/5/29
# Page        : P95 - P96
# ******************************************************************************

import os
import sys

# パス操作
# --- カレントディレクトリの指定
# --- システムパスの追加
cd = os.path.join(os.getcwd(), os.path.join("book", "dl_system", "chap2", "2-06_data-expansion"))
os.chdir(cd)
sys.path.append(cd)

# モジュール
from P01_model_maker import ModelMaker


# パラメータ設定
# --- REUSE_CNTを追加
SRC_DIR = 'D00_dataset/training'
DST_DIR = 'D01_estimator'
EST_FILE = os.path.join(DST_DIR, 'estimator.h5')
INFO_FILE = os.path.join(DST_DIR, 'model_info.txt')
GRAPH_FILE = os.path.join(DST_DIR, 'model_graph.pdf')
HIST_FILE = os.path.join(DST_DIR, 'history.pdf')
INPUT_SIZE = (160, 160)
FILTERS = (64, 128, 256)
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_DIMS = [1024, 128]
LR = 1e-3
BATCH_SIZE = 32
REUSE_CNT = 3
EPOCHS = 5
VALID_RATE = 0.2

# 訓練データのカテゴリ数
# --- フォルダ数
# --- 全結合層の末尾に出力層を追加する処理(クラス数がユニット数になる)
n_class = len(os.listdir(SRC_DIR))
DENSE_DIMS.append(n_class)


# モデル定義
maker = ModelMaker(
    src_dir=SRC_DIR,
    dst_dir=DST_DIR,
    est_file=EST_FILE,
    info_file=INFO_FILE,
    graph_file=GRAPH_FILE,
    hist_file=HIST_FILE,
    input_size=INPUT_SIZE,
    filters=FILTERS,
    kernel_size=KERNEL_SIZE,
    pool_size=POOL_SIZE,
    dense_dims=DENSE_DIMS,
    lr=LR,
    batch_size=BATCH_SIZE,
    reuse_count=REUSE_CNT,
    epochs=EPOCHS,
    valid_rate=VALID_RATE
)

# モデル実行
maker.execute()
