import re
basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
with open(basepath+"uk_articles.csv","r") as f:
    lines = f.readlines()
f.close()

pat_22 = r"\[\[Category:(.*)]\]"#(.*)：()でグルーピングされた任意の文字列を抽出できる．
for l in lines:
    if re.search(pat_22,l):
        print(re.search(pat_22,l).group(1))#group(1)：マッチした文字列の取得．(0)だと文章全体を返す