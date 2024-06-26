#No23(セクション構造)
#今回の問題でいうレベルは、＝の総数\2-1で求められる
import pandas as pd
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
#=の連なりから始まって何か文字が入り、その後に＝の連なりで終わる文字列を探す
for text in D_8[0].split("\n"):
    if re.search("^=+.*=+$", text):
        num = text.count("=") / 2 - 1
        print(text.replace("=", ""), int(num))
#「^」　ここが先頭
#「$」　ここが末尾
#「〇+」〇を一回以上繰り返す（「〇*」〇を０回以上繰り返す）
#「.」　何かの文字を表す