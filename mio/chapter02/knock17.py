#問17
#1列目の文字列の種類（異なる文字列の集合）を求めよ．
#確認にはcut, sort, uniqコマンドを用いよ

#DataFrameの1列目を抽出→重複を無くす

#set()関数（組み込み）：引数にイテラブルオブジェクトを受け取る
#　　　　　　　　　　　 戻り値はsetオブジェクト(注意：setオブジェクトの要素は順序と重複がない)

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

print(set(df.iloc[:, 0]))

#UNIXコマンド
command = ["cut", "-f num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt| sort | uniq"]
subprocess.check_output(command) 

