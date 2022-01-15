# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 1 PyTorchの使い方
# Theme       : PyTorchのコマンド(1-3)
# Creat Date  : 2022/1/15
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - Pytorchの配列作成や演算などのベースコマンドを確認する


# ＜目次＞
# 0 準備
# 1 PyTorchのコマンド1（テンソルの定義）
# 2 PyTorchのコマンド2（テンソルの演算）
# 3 PyTorchのコマンド3（テンソルの操作）


# 0 準備 ------------------------------------------------------------------------

import torch
import numpy as np


# 1 PyTorchのコマンド1（テンソルの定義）-----------------------------------------------

# 空の配列の作成
x = torch.empty(5, 3)
print(x)

# 乱数の生成
x = torch.rand(5, 3)
print(x)

# ゼロ配列の作成
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# テンソルの作成
torch.tensor([1, 23, 3])

# 元の配列サイズを維持して乱数を生成
# --- 1の配列を作成
# --- 元の配列を正規乱数に変換
x = x.new_ones(5, 3, dtype=torch.float)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 数列を生成
# --- 区間指定でステップを設定して数列を作成
x = torch.arange(0, 19, 2)
print(x)

# 数列を生成
# --- 指定区間を指定した要素数に均等分割した数列を作成
x = torch.linspace(0, 3, 10)
print(x)

# テンソルのデータ型を確認
x = torch.FloatTensor([5, 6, 7])
print(x)
print(x.dtype)

# データ型を指定してテンソル作成
# --- 整数型でテンソルを作成
x = torch.tensor([5, 6, 7], dtype=torch.int)
print(x)
print(x.dtype)

# 乱数シードを用いて乱数生成
# --- 直後の実行に対してシードが適用される
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)

# --- 2回目の実行（シードが適用されていない）
x = torch.rand(2, 3)
print(x)

# --- シードを設定して3回目の実行
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)

# テンソルのサイズ
x = torch.rand(2, 3)
print(x.size())
print(x.shape)


# 2 PyTorchのコマンド2（テンソルの演算）-----------------------------------------------

# 演算用のデータ作成
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x)

# 行列の加算
print(x + y)

# 行列の加算
# --- 空の配列の作成
# --- 空の配列に結果を格納
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 行列の加算
# --- 元の配列(y)を上書き
y.add_(x)
print(y)

# 配列のスライス
# --- 列の首都億
# --- 行の取得
# --- 行列の取得
print(y[:, 1])
print(y[0, :])
print(y[2:, 1:])

# 配列の変形
# --- 16個の要素の配列
# --- -1を指定すると、8に対する残りの行列数を自動指定
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# テンソルを数値オブジェクトに変換
# --- torchの出力結果はテンソルオブジェクト
# --- 数値型に変換して出力
x = torch.randn(1)
print(x)
print(x.item())
print(type(x.item()))


# 3 PyTorchのコマンド3（テンソルの操作）-----------------------------------------------

# torch ⇒ numpy ---------------------------------------------

# データ作成
# --- テンソルオブジェクト
a = torch.ones(3)
print(a)
print(type(a))

# numpyオブジェクトに変換
b = a.numpy()
print(b)
print(type(b))

# テンソルの加算（オブジェクトの更新）
# --- aとbはメモリを共有するため、bの値も更新される
a.add_(3)
print(a)
print(b)


# numpy ⇒ torch ---------------------------------------------

# データ作成
# --- numpyオブジェクト
a = np.ones(3)
print(a)

# テンソルオブジェクトに変換
b = torch.from_numpy(a)
print(b)


# torchのメソッド --------------------------------------------

# データ作成
a = torch.arange(1, 4, 1, dtype=torch.float)

# 平均
a.mean()

# 合計
a.sum()

# 最大 / 最小
a.max()
a.min()

# メソッドのチェーン
a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([4, 5, 6], dtype=torch.float)
torch.add(a, b).sum()


# 三角関数の操作 ---------------------------------------------

import math

a = torch.tensor([math.radians(30), math.radians(60), math.radians(90)])
print(torch.sin(a))
print(torch.cos(a))
print(torch.tan(a))
print(torch.asin(a))
print(torch.tanh(a))


# 行列の掛け算 -----------------------------------------------

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
b = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
torch.mm(a, b)
a@b