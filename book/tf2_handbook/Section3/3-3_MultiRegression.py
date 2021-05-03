# ******************************************************************************
# Title     : TensorFlow2プログラミング実装ハンドブック
# Chapter   : 3 TensorFlowの基本
# Theme     : 3-3 tf.GradientDescentOptimizerクラスで多項式の回帰問題を解く
# Created by: Owner
# Created on: 2020/12/2
# Page      : P106 - P112
# ******************************************************************************


# ＜目的＞
# - tf.勾配降下アルゴリズムによるパラメータ最適化を実装して多項式の回帰問題を解く
# - 多項式でモデルを定義することで分布によりフィットするモデルを構築する


# ＜ポイント＞
# - tensorflowにはテンソルを扱うための数学関数が一通り揃っている
#   --- 計算結果はTensorオブジェクトで返される


# ＜目次＞
# 1 準備
# 2 回帰モデルと損失関数の作成
# 3 勾配降下アルゴリズムによる最適化処理の実装
# 4 プロット作成


# 1 準備 ---------------------------------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path


# 関数定義
# --- データ基準化
def standardize(x):
    x_mean = x.mean()
    std = x.std()
    return (x - x_mean) / std


# データ取り込み
path = Path("book/tf2_handbook/csv/sales.csv")
data = np.loadtxt(fname=path, dtype='int', delimiter=',', skiprows=1)

# アイテム取得
train_x = data[:, 0]
train_y = data[:, 1]

# 基準化
train_x_std = standardize(train_x)
train_y_std = standardize(train_y)


# 2 回帰モデルと損失関数の作成 -------------------------------------------------------------------

# 変数定義
# --- バイアス(b): 切片
# --- ウエイト(w): 傾き
b = tf.Variable(0.)
w1 = tf.Variable(0.)
w2 = tf.Variable(0.)
w3 = tf.Variable(0.)
w4 = tf.Variable(0.)


# 回帰モデルの定義
def model(x):
    y = b + w1 * x + w2 * pow(x, 2) + w3 * pow(x, 3) + w4 * pow(x, 4)
    return y


# 損失関数の定義
# --- MSE(平均二乗誤差)
def loss(y_pred, y_true):
    """ MSE(平均二乗誤差)

    Parameters:
      y_pred(ndarray): 予測値
      y_true(ndarray): 正解値
    """
    return tf.math.reduce_mean(tf.math.square(y_pred - y_true))


# 3 勾配降下アルゴリズムによる最適化処理の実装 ----------------------------------------------------

# パラメータ設定
# --- 学習率の設定（少し細かめに設定）
# --- 学習回数
learning_rate = 0.01
epochs = 200

# 勾配降下アルゴリズム
i = 1
for i in range(epochs):
    # 勾配を計算するブロック
    with tf.GradientTape() as tape:
        y_pred_std = model(train_x_std)
        tmp_loss = loss(y_pred_std, train_x_std)

    # 勾配計算
    # --- tapeに記録されてた操作を使用
    gradients = tape.gradient(target=tmp_loss, sources=[b, w1, w2, w3, w4])

    # パラメータ更新
    # --- 勾配降下法の更新式を適用
    b.assign_sub(learning_rate * gradients[0])
    w1.assign_sub(learning_rate * gradients[1])
    w2.assign_sub(learning_rate * gradients[2])
    w3.assign_sub(learning_rate * gradients[3])
    w4.assign_sub(learning_rate * gradients[4])

    # 結果出力
    # --- 50回ごとに出力
    if (i + 1) % 50 == 0:
        # 処理回数(i), ウエイト(w), 切片(b)
        print('Step:{}\n a1 = {}\n a2 = {}\n a3 = {}\n a4 = {}\n b = {}\n'.format(
            i + 1,
            w1.numpy(),
            w2.numpy(),
            w3.numpy(),
            w4.numpy(),
            b.numpy())
        )

        # 損失を出力
        print('Loss = {}'.format(tmp_loss))


# 4 プロット作成 ----------------------------------------------------------------------------

# 散布図
plt.scatter(train_x_std, train_y_std)

# 等差数列の作成
# --- プロットのX軸に使用
x_axis = np.linspace(start=-2, stop=2, num=100)

# 予測値の取得
y_learned = b + w1 * x_axis + w2 * pow(x_axis, 2) + \
            w3 * pow(x_axis, 3) + w4 * pow(x_axis, 4)

# プロット作成
plt.plot(x_axis, y_learned, 'r')
plt.grid(True)
plt.show()
