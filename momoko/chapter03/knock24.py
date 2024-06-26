#https://ja.wikipedia.org/wiki/Help:%E7%94%BB%E5%83%8F%E3%81%AA%E3%81%A9%E3%81%AE%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%81%AE%E3%82%A2%E3%83%83%E3%83%97%E3%83%AD%E3%83%BC%E3%83%89%E3%81%A8%E5%88%A9%E7%94%A8
#[[ファイル:tst.png]]
import re
basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
with open(basepath+"uk_articles.csv","r") as f:
    lines = f.readlines()
f.close()

pat ='\[\[ファイル:(.*?)(?:\||\])'#(.*)：()でグルーピングされた部分を抜く
for l in lines:
    if re.search(pat,l):
        print(re.search(pat,l).group(1))#group(1)：マッチした文字列のみ取る