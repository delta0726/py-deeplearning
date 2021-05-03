# ******************************************************************************
# Title     : TensorFlow2プログラミング実装ハンドブック
# Chapter   : 3 TensorFlowの基本
# Theme     : 3-4 パーセプトロンによる二値分類
# Created by: Owner
# Created on: 2020/12/10
# Page      : P113 - P127
# ******************************************************************************


# ＜目的＞
# - 勾配降下アルゴリズムによるパラメータ最適化を実装して分類問題を解く


# ＜ポイント＞
# - tensorflowにはテンソルを扱うための数学関数が一通り揃っている
#   --- 計算結果はTensorオブジェクトで返される


# ＜目次＞
# 1 準備
# 2 分類関数の定義
# 3 ウエイトの更新処理を実装
# 4 パーセプトロンによる二値分類の実行
# 5 プロット作成


# 1 準備 ---------------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# データ取り込み
path = Path("book/tf2_handbook/csv/negaposi.csv")
data = np.loadtxt(fname=path, dtype='int', delimiter=',', skiprows=1)

# データ取得
# --- x: 特徴量
# --- y: ラベル(1 or -1)
x = data[:, 0:2]
t = data[:, 2]

# 散布図の作成
# --- ラベル種別で色付け（二値分類）
plt.plot(x[t == 1, 0], x[t == 1, 1], 'o')
plt.plot(x[t == -1, 0], x[t == -1, 1], 'x')
plt.show()


# 2 分類関数の定義 --------------------------------------------------------------------------------

# 関数定義
# --- 2値分類
def classify(x, w):
    """パーセプトロン（分類関数）

       Parameters:
         x(ndarray): x1、x2のデータ
         w(ndarray): w1、w2の値
       Returns:
         (float)更新後のウエイト w1、w2
    """
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1


# 3 ウエイトの更新処理を実装 -----------------------------------------------------------------------

def learn_weight(x, t):
    """更新式で重みを学習する

       Parameters:
         x(ndarray): x1、x2のデータ
         w(ndarray): t(正解ラベル)
       Returns:
         (int)更新後のw(重み)
    """
    # 乱数シード
    np.random.seed(seed=1)

    # パラメータ設定
    w = np.random.rand(2)
    epochs = 5
    count = 0

    # 指定した回数だけウエイト更新を繰り返す
    for i in range(epochs):
        # ベクトルx、tから成分を取り出す
        for element_x, element_t in zip(x, t):
            # 分類関数の出力が異なる場合は重みを更新する
            if classify(element_x, w) != element_t:
                w = w + element_t * element_x
                print('更新後のw = ', w)
        count += 1
        # ログの出力
        print('[{}回目]: w = {}***'.format(count, w))
    return w


# 4 パーセプトロンによる二値分類の実行 ----------------------------------------------------------

# 学習
w = learn_weight(x, t)


# 5 プロット作成 -----------------------------------------------------------------------------

# 軸の範囲を設定
x1 = np.arange(0, 600)

# プロット作成
plt.plot(x[t == 1, 0], x[t == 1, 1], 'o')
plt.plot(x[t == -1, 0], x[t == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='solid')
plt.show()
