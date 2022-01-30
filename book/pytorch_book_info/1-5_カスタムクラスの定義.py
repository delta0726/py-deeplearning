# ******************************************************************************
# Book        : 最短コースでわかる PyTorch ＆深層学習プログラミング
# Chapter     : 1 ディープラーニングのためのPythonのツボ
# Theme       : 5 カスタムクラスの定義
# Creat Date  : 2022/1/30
# Final Update:
# URL         : https://github.com/makaishi2/pytorch_book_info
# Page        : P42 - P54
# ******************************************************************************


# ＜概要＞
# - カスタムクラスとはユーザーが目的に応じて独自にクラスを定義したものをいう
# - Pytorchはカスタムクラスを定義することでモデル構築を行う
#   --- オブジェクト指向プログラミングの基本が分かっている必要がある


# ＜目次＞
# 0 準備
# 1 最初のクラス定義
# 2 Circleクラスの定義(継承先で再定義)
# 3 Circleクラスの定義(継承元で機能追加)


# 0 準備 ------------------------------------------------------------

# ライブラリ
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 1 最初のクラス定義 -------------------------------------------------

# ＜ポイント＞
# - クラス定義とインスタンス生成の流れを確認する
#   --- コンストラクタ/メソッド/インスタンスなどの概念を再確認


# クラス定義
class Point:
    def __init__(self, x, y):
        """
        インスタンス生成時の初期処理
        --- ここでは受け取った引数をselfに格納
        """
        self.x = x
        self.y = y
    
    def draw(self):
        """
        メソッド
        --- selfに格納されたxとyでプロットを描く処理
        """
        plt.plot(self.x, self.y, marker='o', markersize=10, c='k')


# インスタンス生成
p1 = Point(x=2, y=3)
p2 = Point(x=-1, y=-2)

# プロパティ確認
# --- インスタンスで与えた引数はプロパティとして取得可能
print(p1.x, p1.y)
print(p2.x, p2.y)

# プロット作成
# --- p1とp2のdrawメソッド(関数)を呼び出して2つの点を描画する
p1.draw()
p2.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# 2 Circleクラスの定義(継承先で再定義) -------------------------------

# ＜ポイント＞
# - 継承元のdrawメソッドを継承先で再定義する
#   --- 再定義された操作のみが行われる（クラス継承した恩恵がない）


# クラス定義
# --- Pointクラスを継承
class Circle2(Point):
    def __init__(self, x, y, r):
        """
        xとyは上位クラスでselfに格納されている
        ここでは新しく定義したrのみを格納する
        """
        super().__init__(x, y)
        self.r = r

    def draw(self):
        """
        drawメソッドを再定義
        円を表示する
        """
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)


# インスタンスの生成
c2_1 = Circle2(x=1, y=0, r=2)

# プロット作成
# --- 再定義したdrawメソッドが実行されるので円が描かれる
ax = plt.subplot()
c2_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()


# 3 Circleクラスの定義(継承元で機能追加) -----------------------------

# ＜ポイント＞
# - 継承元のdrawメソッドを実行したうえで、継承先で独自の処理を追加する
#   --- 継承元のメソッドに新たな機能を追加（継承の恩恵を活かしている）


class Circle3(Point):
    def __init__(self, x, y, r):
        """
        xとyは上位クラスでselfに格納されている
        ここでは新しく定義したrのみを格納する
        """
        super().__init__(x, y)
        self.r = r

    def draw(self):
        """
        drawメソッドを再定義
        継承元のdrawを実行してから、円を表示する
        """
        super().draw()
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)


# インスタンスの生成
c3_1 = Circle3(x=1, y=0, r=2)

# プロット作成
# --- 継承元の点表示と、継承先で定義した円表示の両方が実行されている
ax = plt.subplot()
c3_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
