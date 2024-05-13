#問13
#12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．
#確認にはpasteコマンドを用いよ．

#方針１：問12で作成した２つのcsvファイルを呼び出す
#方針２：横（行方向）に連結
#方針３：タブ区切りでファイルに書き込む
#方針４：pasteコマンドで確認

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

#作業１　read_csv()メソッド
col1 = pd.read_csv("col1.txt", header=None)
col2 = pd.read_csv("col2.txt", header=None)

#作業２　pd.concat([DataFrame,DataFrame])関数：２つのデータフレームを結合（※concatenate鎖状につなぐ）
#　　　　　　　　　　　　　　　　　　　　　　　行方向(横)への結合→axis=1、列方向(縦)への結合→ axis=0
col1_2 = pd.concat([col1, col2], axis=1)

#作業３　to_csvメソッド（引数sep：区切り文字指定）
col1_2.to_csv("col1_2.txt", sep="\t", header=False, index=False)

#作業４　paste ファイル1 ファイル2...>連結後のデータを格納するファイル:複数のファイルを行単位で連結
command = ["paste", "col1.txt col2.txt > col1_2.txt"]
subprocess.check_output(command) 
