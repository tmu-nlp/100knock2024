#1列目をcol1.txtに，2列目をcol2.txtに保存
#各行の1列目だけを抜き出したものをcol1.txtに，
#2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
#確認にはcutコマンドを用いよ．

#dfから１列目、２列目のデータを抽出し、変数df1,df2にそれぞれ保存する。

df1 = df.iloc[:, 0] 
df2 = df.iloc[:, 1]

df1.to_csv("col1.txt", sep=",", header=False, index=False)
df2.to_csv("col2.txt", sep=",", header=False, index=False)