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

#.は文字*は繰り返すこと?は最小単位で区切る
pattern = "\[\[Category:.*?\]\]"
#re.fingall(正規表現パターン,文字列)
result = re.findall(pattern, UK_text)
print(result)