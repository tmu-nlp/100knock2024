import re

#https://ja.wikipedia.org/wiki/Help:%E3%82%AB%E3%83%86%E3%82%B4%E3%83%AA
#によると，[[Category:弦楽器]]or[[カテゴリ:弦楽器]]の形式で書かれているらしい!!

basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
with open(basepath+"uk_articles.csv","r") as f:
    lines = f.readlines()
f.close()

pat = "\[\[Category:.*?\]\]"#.*?：任意の1文字,0回以上の繰り返し，0,1回 #\[：特殊文字でなく文字そのものとして識別する
for l in lines:
    if re.match(pat,l):#patとlがマッチしたら
        print(l.replace("\n",""))#改行を削除しつつprint