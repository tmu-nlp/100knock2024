import json
import re
text_list = [] #空のリストを用意
with open('./jawiki-country.json', 'r') as f: #カレントディレクトリのjawiki-country.jsonファイルをfとして読み取る
    lines = f.readlines() #ファイルの文字列を1行ごとにをリストに入れる
    for line in lines:
        text_list.append(json.loads(line)) #json.loadsは文字列を引数にとり、辞書型にして返す。それをtext_listに追加

for i in range(len(text_list)):
    if text_list[i]['title']=="イギリス": #リストtext_listのi番目のキーがtitleのものについて、値がイギリスかどうか
        UK_text = str(text_list[i])
        break

pattern = '基礎情報(.*?\<references/\>)'
result = re.findall(pattern, UK_text)
result[0] += "\\n"
pattern = '(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)'
result2 = re.findall(pattern, result[0])
inf_dic = {}
for i, j in result2:
  inf_dic[i] = j

inf_dic2 = {}
for key, text in inf_dic.items():
  inf_dic2[key] = re.sub(r'(\\\'){2,5}' , '', text)
print(inf_dic2)

#以下でも似たことができる

# import pandas as pd
# import re
# pattern = re.compile('\|(.+?)\s=\s*(.+)')
# p_emp = re.compile('\'{2,}(.+?)\'{2,}')
# wiki = pd.read_json('jawiki-country.json.gz', lines = True)
# uk = wiki[wiki['title']=='イギリス'].text.values
# ls = uk[0].split('\n')
# d = {}
# for line in ls:
#     r = re.search(pattern, line)
#     if r:
#         d[r[1]]=r[2]
#     r = re.sub(p_emp,'\\1', line)
#     print(r)
# print(d)