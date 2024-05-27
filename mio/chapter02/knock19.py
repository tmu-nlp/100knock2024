#問19
#各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ
#確認にはcut, uniq, sortコマンドを用いよ

#方針１：DataFrameの1列目を取得
#方針２：value_counts()で求められた１列目の各要素数を降順に並び替え

#Series.value_counts()メソッド：デフォルトで、各要素数を降順で出力する。
#DataFrameに使用する場合は、DataFrame.apply()メソッドと併用する必要がある。

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)


sr = df[0]
print(sr.value_counts())

#UNIXコマンド
command = ["cut", "-f num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt| sort | uniq-c | sort -n -r"]
subprocess.check_output(command) 
