# ******************************************************************************
# Course      : PyTorch Boot Camp : Python AI PyTorchで機械学習とデータ分析完全攻略
# Chapter     : 4 CNN（畳み込みニューラルネットワーク）
# Theme       : CNNによるMNISTの学習
# Creat Date  : 2022/1/23
# Final Update:
# URL         : https://www.udemy.com/course/python-pytorch-facebookai/
# ******************************************************************************


# ＜概要＞
# - CNNは行列の位置に対してロバスト性を持たせるもので、画像処理において効果を発揮する技術


# ＜公式＞
# (InputSize - FilterSize + 2P) / S + 1
# P: パディング
# S: ストライド


# ＜目次＞
# 0 準備


# 0 準備 -------------------------------------------------------------------

# ライブラリ
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# torchvisionのインストールでエラー発生