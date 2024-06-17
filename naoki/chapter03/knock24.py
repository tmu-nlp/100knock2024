import json
import gzip
import re

text_list = []
with gzip.open('./jawiki-country.json.gz') as f:
    lines = f.readlines()
    for line in lines:
        #json.loads:json形式をデコード
        #text_list[1]はエジプトの記事
        text_list.append(json.loads(line)) 
#どこかにあるイギリスの記事を抽出
for i in range(len(text_list)):
    if text_list[i]['title']=="イギリス":
        UK_text = str(text_list[i])  
#ファイル:…がウィキペディアのマークアップ言語で画像を示している
#[[ファイル: で始まり、| または ] で終わる文字列を検索
pattern = '\[\[ファイル:(.*?)(?:\||\])'
result = re.findall(pattern, UK_text)
print(result)