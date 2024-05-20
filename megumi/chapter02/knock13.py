#col1.txtとcol2.txtをマージ
#12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ．
import pandas as pd

col1 = pd.read_csv("col1.txt", header=None)
col2 = pd.read_csv("col2.txt", header=None)

col1_2= pd.concat([col1, col2], axis=1)
col1_2.to_csv("col1_2.txt", sep="\t", header=False, index=False)
