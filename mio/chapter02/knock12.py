#問12
#各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
#確認にはcutコマンドを用いよ．

#DataFrame.iloc[行,列]メソッド：データ抽出

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

#dfから1列目のデータを抽出し変数df1に保存
df1 = df.iloc[:, 0]

#dfから2列目のデータを抽出し、変数df2に保存
df2 = df.iloc[:, 1]
 

#DataFrame.to_csv(ファイル名)メソッド：csvファイルに書き出す
#　　　　　　　　　　　　　　　　　　　headerとindexは無しにしたいのでオプションでFalseに
#　　　　　　　　　　　　　　　　　　　引数sep：ファイル内のデータの区切り文字を指定

df1.to_csv("col1.txt", sep=",", header=False, index=False)
df2.to_csv("col2.txt", sep=",", header=False, index=False)