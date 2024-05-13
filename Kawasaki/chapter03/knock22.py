import json
import re
text_list = [] #空のリストを用意
with open('./jawiki-country.json', 'r') as f: #カレントディレクトリのjawiki-country.jsonファイルをfとして読み取る
    lines = f.readlines() #ファイルの文字列を1行ごとにをリストに入れる
    for line in lines:
        text_list.append(json.loads(line)) #json.loadsは文字列を引数にとり、辞書型にして返す。それをtext_listに追加

for i in range(len(text_list)):
    if text_list[i]['title']=="イギリス": #リストtext_listのi番目のキーがtitleのものについて、値がイギリスかどうか
        UK_text = str(text_list[i]['text'])
        break

pattern = "\[\[Category:(.*?)(?:\|.*?|)\]\]"
result = re.findall(pattern, UK_text)
print(result)

#以下でも似たことができる

# import pandas as pd
# import re
# wiki = pd.read_json('jawiki-country.json.gz', lines = True)
# UK_text = wiki[wiki['title']=='イギリス'].text.values
# ls = uk[0].split('\n')
# for line in ls:
#     if re.search(pattern, line):
#         line = line.replace('[[','').replace('Category:','').replace(']]','').replace('|*','').replace('|元','')
#         print (line)