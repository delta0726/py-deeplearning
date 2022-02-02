# ******************************************************************************
# Book        : 最短コースでわかる PyTorch ＆深層学習プログラミング
# Chapter     : 3 初めての機械学習
# Theme       : 5 はじめての機械学習プロセス
# Creat Date  : 2022/1/31
# Final Update:
# URL         : https://github.com/makaishi2/pytorch_book_info
# Page        : P100 - P129
# ******************************************************************************


# ＜概要＞
# - 勾配計算機能を使って線形回帰の機械学習問題を解く
#   --- 勾配降下法の考え方はディープラーニングと同じ


# ＜勾配降下法＞
# - ｢予測計算｣｢損失計算｣｢勾配計算｣｢パラメータ修正｣を繰り返して行うことで学習を行う
#   --- P105の概念図が分かりやすい


# ＜目次＞
# 0 準備
# 1 データ定義
# 2 データ前処理
# 3 予測計算
# 4 損失計算
# 5 勾配計算
# 6 パラメータ修正
# 7 繰り返し計算
# 8 学習曲線の確認
# 9 散布図と回帰直線の確認
# 10 最適化関数とstep関数の利用


# 0 準備 -----------------------------------------------------------------

# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch
import torch.optim as optim
from torchviz import make_dot

# 1 データ定義 -----------------------------------------------------------

# ＜ポイント＞
# - 身長から体重を予測するモデルを構築する
#   --- データはP100-P101を参照


# データ生成
sampleData1 = np.array([
    [166.0, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])

# 確認
print(sampleData1)

# データ定義
x = sampleData1[:,0]
y = sampleData1[:,1]

# プロット作成
plt.scatter(x,  y,  c='k',  s=50)
plt.xlabel('$x$: 身長(cm) ')
plt.ylabel('$y$: 体重(kg)')
plt.title('身長と体重の関係')
plt.show()


# 2 データ前処理 ---------------------------------------------------------

# ＜ポイント＞
# - データ基準化の方法として中心化を行う
#   --- Zスコア変換のほうがスッキリするが、xの特徴量は1つなので中心化で問題ない

# 中心化
X = x - x.mean()
Y = y - y.mean()

# プロット作成
plt.scatter(X,  Y,  c='k',  s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('加工後の身長と体重の関係')
plt.show()


# 3 予測計算 --------------------------------------------------------------

# ＜ポイント＞
# - 予測値を算出するためのモデルを構築して予測値を求める
#   --- この時点でパラメータとバイアスは未知


# 関数定義
# --- Y = W * X + B
def pred(X):
    return W * X + B


# データ変換
# --- 特徴量とラベルをテンソル変換
Y = torch.tensor(Y).float()
X = torch.tensor(X).float()

# 確認
print(X)
print(Y)

# パラメータ初期値
# --- ウエイト(W)とバイアス(B)
# --- 自動微分の勾配計算をするので、requires_grad=Trueとする
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 予測値の計算
# --- パラメータ初期値から算出した予測値
Yp =  pred(X)

# 確認
print(Yp)

# # 計算グラフ可視化
# # --- 予測値
# params = {'W': W, 'B': B}
# g = make_dot(Yp, params=params)
# g


# 4 損失計算 ---------------------------------------------------

# ＜ポイント＞
# - 今回は損失関数として平均二乗誤差(MSE)を用いる
#   --- この損失量を減らすようにパラメータ修正を行っていく


# 関数定義
# --- 平均二乗誤差
def mse(Yp, Y):
    loss = ((Yp - Y) ** 2).mean()
    return loss

# 損失計算
loss = mse(Yp, Y)

# 結果標示
print(loss)

# 損失の計算グラフ可視化
# params = {'W': W, 'B': B}
# g = make_dot(loss, params=params)


# 5 勾配計算 ----------------------------------------------------

# ＜ポイント＞
# - 現在のバイアスとパラメータにおける勾配を計算する
#   --- パラメータ修正の準備（勾配が小さくなるほうにパラメータを変更）


# 勾配計算
loss.backward()

# 勾配値の確認
print(W.grad)
print(B.grad)


# 6 パラメータ修正 ------------------------------------------------

# ＜ポイント＞
# - 勾配に基づくパラメータ修正の度合いは学習率によってコントロールする
# - 勾配を計算している途中の変数(WとB)は勝手に値を変更することができない
#   --- with文を定義することで一時的に変更可能となる
#   --- 実際にはオプティマイザーを使うことでエレガントに記述することが可能


# 学習率
lr = 0.001

# 勾配を元にパラメータ修正
# --- with torch.no_grad() を付ける必要がある
with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad

    # 計算済みの勾配値をリセットする
    W.grad.zero_()
    B.grad.zero_()

# 確認
# --- パラメータと勾配値
print(W)
print(B)
print(W.grad)
print(B.grad)


# 7 繰り返し計算 -----------------------------------------------------

# ＜ポイント＞
# - これまで行った｢予測計算｣｢損失計算｣｢勾配計算｣｢パラメータ修正｣を連続で行う
#   --- イテレーションを回して学習を進める


# パラメータ初期値
# --- ウエイト(W)とバイアス(B)
# --- 自動微分の勾配計算をするので、requires_grad=Trueとする
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# その他の設定
# --- 繰り返し回数
# --- 学習率
num_epochs = 500
lr = 0.001

# 配列の初期化
# --- 記録用
history = np.zeros((0, 2))

# ループ処理
# epoch = 0
for epoch in range(num_epochs):

    # 学習プロセス
    # --- 予測計算 ⇒ 損失計算 ⇒ 勾配計算
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()

    # パラメータ修正/勾配初期化
    with torch.no_grad():
        W -= lr * W.grad
        B -= lr * B.grad

        W.grad.zero_()
        B.grad.zero_()

    # 損失の記録
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')


# 8 学習曲線の確認 --------------------------------------------------------

# ＜ポイント＞
# - イテレーションを回すタイプの学習では学習曲線が出力される
#   --- ディープラーニングや勾配ブースティング


# パラメータの最終値
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

# 損失の確認
print(f'初期状態: 損失:{history[0,1]:.4f}')
print(f'最終状態: 損失:{history[-1,1]:.4f}')

# プロット作成
# --- 学習曲線の表示 (損失)
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()


# 9 散布図と回帰直線の確認 --------------------------------------------------

# ＜ポイント＞
# - 今回の学習の目的は線形回帰だったので、回帰直線が正しく作成されているか確認


# xの範囲の算出
X_max = X.max()
X_min = X.min()
X_range = np.array((X_min, X_max))
X_range = torch.from_numpy(X_range).float()
print(X_range)

# 対応するyの予測値を求める
Y_range = pred(X_range)
print(Y_range.data)

# プロット作成
# --- 加工後の散布図
plt.scatter(X,  Y,  c='k',  s=50)
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.plot(X_range.data, Y_range.data, lw=2, c='b')
plt.title('身長と体重の相関直線(加工後)')
plt.show()


# y座標値とx座標値の計算
x_range = X_range + x.mean()
yp_range = Y_range + y.mean()

# プロット作成
# --- 加工前の散布図
plt.scatter(x,  y,  c='k',  s=50)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(x_range, yp_range.data, lw=2, c='b')
plt.title('身長と体重の相関直線(加工前)')
plt.show()


# 10 最適化関数とstep関数の利用 ----------------------------------------

# パラメータ初期値
# --- ウエイト(W)とバイアス(B)
# --- 自動微分の勾配計算をするので、requires_grad=Trueとする
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# その他の設定
# --- 繰り返し回数
# --- 学習率
num_epochs = 500
lr = 0.001

# オプティマイザーの指定
# --- SGD(確率的勾配降下法)を指定する
optimizer = optim.SGD([W, B], lr=lr)

# 配列の初期化
# --- 記録用
history = np.zeros((0, 2))

for epoch in range(num_epochs):

    # 学習プロセス
    # --- 予測計算 ⇒ 損失計算 ⇒ 勾配計算 ⇒ パラメータ修正 ⇒ 勾配値初期化
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 損失値の記録
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')


# パラメータの最終値
print('W = ', W.data.numpy())
print('B = ', B.data.numpy())

#損失の確認
print(f'初期状態: 損失:{history[0,1]:.4f}')
print(f'最終状態: 損失:{history[-1,1]:.4f}')

# プロット作成
# --- 学習曲線の表示 (損失)
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.show()


# 11 最適化関数のチューニング -------------------------------------------

# パラメータ初期値
# --- ウエイト(W)とバイアス(B)
# --- 自動微分の勾配計算をするので、requires_grad=Trueとする
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# その他の設定
# --- 繰り返し回数
# --- 学習率
num_epochs = 500
lr = 0.001

# オプティマイザーの指定
# --- SGD(確率的勾配降下法)を指定する
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)

# 配列の初期化
# --- 記録用
history2 = np.zeros((0, 2))

# ループ処理
for epoch in range(num_epochs):

    # 学習プロセス
    # --- 予測計算 ⇒ 損失計算 ⇒ 勾配計算 ⇒ パラメータ修正 ⇒ 勾配値初期化
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 損失値の記録
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history2 = np.vstack((history2, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')


# プロット作成
# --- 学習曲線の表示 (損失)
plt.plot(history[:,0], history[:,1], 'b', label='デフォルト設定')
plt.plot(history2[:,0], history2[:,1], 'k', label='momentum=0.9')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.legend()
plt.title('学習曲線(損失)')
plt.show()
