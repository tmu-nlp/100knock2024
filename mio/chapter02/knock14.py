#問14
#自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．
#確認にはheadコマンドを用いよ．

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)


num = int(input())

#DataFrame.head()メソッド(Series.head())：デフォルトは先頭5行を出力
df.head(num)

#UNIXコマンド
command = ["head", "-n num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt"]
subprocess.check_output(command) 