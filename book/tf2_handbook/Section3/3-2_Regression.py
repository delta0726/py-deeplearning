# ******************************************************************************
# Title     : TensorFlow2プログラミング実装ハンドブック
# Chapter   : 3 TensorFlowの基本
# Theme     : 3-2 tf.GradientTapeクラスで回帰問題を解く
# Created by: Owner
# Created on: 2020/11/25
# Page      : P81 - P105
# ******************************************************************************


# ＜目的＞
# - tf.GradientTapeクラスで勾配降下アルゴリズムによるパラメータ最適化を実装する
#   --- tf.GradientTapeクラスは自動微分のための操作を記録するクラス
#   --- tfクラスの基本関数を確認する

# ＜ポイント＞
# - 線形回帰モデルを使用して、ネット広告費に対する売上高の予測を行う
# - 線形回帰モデルは勾配降下アルゴリズムによって解く


# ＜目次＞
# 1 準備
# 2 データ基準化
# 3 回帰モデルのパーツ
# 4 勾配降下アルゴリズムの実装
# 5 最適化処理の実装
# 6 グラフの描画
# 7 予測


# 1 準備 ---------------------------------------------------------------------------------------

# ライブラリ
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# パス指定
path = Path("csv/sales.csv")

# データ取り込み
# --- Numpyのloadtxt()を用いて読み込み
# --- numpy.ndarrayオブジェクトとして読込まれる
data = np.loadtxt(fname=path, dtype='int', delimiter=',', skiprows=1)

# データ分割
# --- X: ネット広告費
# --- Y: 売上高
train_x = data[:, 0]
train_y = data[:, 1]

# プロット作成
# --- 第3引数はプロットする記号（Markers）
# --- https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
plt.plot(train_x, train_y, 'o')
plt.show()


# 2 データ基準化 --------------------------------------------------------------------------------

# 関数定義
# --- Zスコア化
def standardize(x):
    """標準化を行う
    Parameters:
        x(ndarray): 標準化前のx
    """
    x_avg = x.mean()
    x_std = x.std()
    return (x - x_avg) / x_std


# データ基準化
train_x_std = standardize(train_x)
train_y_std = standardize(train_y)

# プロット作成
plt.plot(train_x_std, train_y_std, 'o')
plt.show()


# 3 回帰モデルのパーツ ---------------------------------------------------------------------------

# ＜勾配降下法＞
# - 目的関数が描く曲線の最小値を、繰り返し計算によって近似値を求める数値計算法
# - 目的関数は予測値と正解値の誤差(損失)を求めることから｢損失関数｣とも呼ばれる
# - 学習率(η)で収束する速さを調整する


# 変数定義
# --- テンソルとして定義
# --- a: ウエイト(傾き)
# --- b: バイアス(切片項)
a = tf.Variable(0.)
b = tf.Variable(0.)


# 関数定義
# --- 回帰モデルの定義
def model(x):
    """ 回帰モデル y = ax + b

    Parameters:
      x(ndarray):分析するデータ
    """
    y = a * x + b
    return y


# 関数定義
# --- 損失関数を平均二乗誤差として定義
def loss(y_pred, y_true):
    """ MSE(平均二乗誤差)

    Parameters:
      y_pred(ndarray): 予測値
      y_true(ndarray): 正解値
    """
    return tf.math.reduce_mean(tf.square(y_pred - y_true))


# 4 勾配降下アルゴリズムの実装 -------------------------------------------------------------

# ＜ポイント＞
# - GradientTapeオブジェクトの生成から計算登録までをwithブロックでまとめる
#   --- tensorflowの公式ガイドより
#   --- 可読性を高める


# 変数定義
x = tf.Variable(3.0)

# 勾配計算
# --- 自動微分による勾配計算を記録
# --- 公式ガイドではGradientTapeオブジェクトの生成から計算までをwithブロックでまとめることを推奨
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x

# 確認
# --- gはオブジェクトの外からも参照可能
g.gradient(y, x).numpy()


# 5 最適化処理の実装 ------------------------------------------------------------------

# パラメータ設定
# --- 学習率
# --- 学習回数
learning_rate = 0.1
epoches = 50

i = 1
# 勾配降下アルゴリズムによる最適化
for i in range(epoches):

    # 自動微分
    # --- 勾配計算を記録
    with tf.GradientTape() as tape:
        y_pred = model(train_x_std)
        tmp_loss = loss(y_pred, train_y_std)

    # 勾配計算
    gradients = tape.gradient(tmp_loss, [a, b])

    # パラメータ更新
    a.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    # 結果出力
    # --- 5回ごと
    if (i + 1) % 5 == 0:
        print('Step:{} a = {} b = {}'.format(
            i + 1,
            a.numpy(),
            b.numpy())
        )

        # 損失を出力
        print('Loss = {}'.format(tmp_loss))


# 6 グラフの描画 ------------------------------------------------------------------

# プロット作成
# --- 散布図の作成
# --- 回帰直線の追加
plt.scatter(train_x_std, train_y_std)
y_learned = a * train_x_std + b
plt.plot(train_x_std, y_learned, 'r')

# プロット表示
plt.grid(True)
plt.show()


# 7 予測 ------------------------------------------------------------------

# インプット値
input_x = 300

# 基準化
# --- 平均値と標準偏差は訓練データのものを使用
# --- xは訓練データと独立した値を指定
x_mean = train_x.mean()
std = train_x.std()
x = (int(input_x - x_mean) / std)

# 回帰式にあてはめ
y = (a * x + b).numpy()

# 標準化前の水準に戻す
y_mean = train_y.mean()
y_std = train_y.std()
y * y_std + y_mean
