#問15
#自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．
#確認にはtailコマンドを用いよ．

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)


num = int(input())
#DataFrame.tail()メソッド(Series.tail())：デフォルトは末尾の5行を出力
df.tail(num)

#UNIXコマンド
command = ["tail", "-n num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt"]
subprocess.check_output(command) 
