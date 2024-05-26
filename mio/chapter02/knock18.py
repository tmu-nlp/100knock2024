#問18
#各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）
#確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）．

#方針：DataFrameの3列目のデータを降順にする→それをもとにすべてのデータを並び替える。

#DataFrame.sort_values()メソッド(Series.sort_values())：第一引数にsortしたい列名を指定する。指定した列データを基準にDataFrame全体をsortする。デフォルトで昇順にする。
#ascending引数：ascending=Falseを指定することで降順に

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

df.sort_values(2, ascending=False)

#UNIXコマンド
command = ["cut", "-f num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt| sort -n -r"]
subprocess.check_output(command) 

