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

pattern = "\[\[Category:(.*?)(?:\|.*?|)\]\]"
#()は指定の箇所のみ抽出で、2つ目は1つ目の条件を満たす者のうちいらないものを削っている。(?:x)で非キャプチャグル―プ \|で|の意味を消す効果がある。なので|x or 無を非キャプチャにしている
result = re.findall(pattern, UK_text)
print(result)