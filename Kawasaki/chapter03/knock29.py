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

inf_dic3 = {}
for key, text in inf_dic2.items():
  pattern = "(?<=\}\}\<br \/\>（)\[{2}"#<br />（フランス語]]:[[Dieu et mon droit|神と我が権利]]）
  text = re.sub(pattern, '', text)

  pattern = "\[{2}.*?\|.*?px\|(?=.*?\]\])"#'[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]',
  text = re.sub(pattern, '', text)

  pattern = "(?<=(\|))\[{2}"
  text = re.sub(pattern, '', text)

  pattern = "(?<=\}{2}（)\[{2}"#スコットランド・ゲール語]]）\n*{{lang|cy
  text = re.sub(pattern, '', text)

  pattern = "(?<=\>（)\[{2}.*?\|"#[[グレートブリテン及びアイルランド連合王国]]成立<br />（1800年合同法]]）
  text = re.sub(pattern, '', text)

  pattern = "(?<=（.{4}).*?\[{2}.*?\)\|" #'[[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）',
  text = re.sub(pattern, '', text)

  pattern = "\[{2}.*?\|"#[[(除去)|]]の処理
  text = re.sub(pattern, '', text)

  pattern = "(\[{2}|\]{2})"#最後に残ったやつを処理
  inf_dic3[key] = re.sub(pattern, '', text)
  
inf_dic4 = {}
for key, text in inf_dic3.items():
  pattern = '\{\{.*?\{\{center\|' #{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}',
  text = re.sub(pattern, '', text)

  pattern = '\{\{.*?\|.*?\|.{2}\|'
  text = re.sub(pattern, '', text) #'{{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}'

  pattern = '\<ref.*?\>.*?\<\/ref\>'#<ref>で囲んでいるもの
  text = re.sub(pattern, '', text)

  pattern = '\<ref.*?\>|\<br \/\>'#<ref>単体+<br />単体
  text = re.sub(pattern, '', text)

  pattern = '\{\{lang\|.*?\|'#公式国名、標語の外国語タイトルを消すため
  text = re.sub(pattern, '', text)

  pattern = '\{\{.*?\}\}'#確立年月日2,3,4の{を除去する
  text = re.sub(pattern, '', text)

  pattern = '\}\}'# 公式国名、標語、国家、他元首等氏名2の}を除去する
  text = re.sub(pattern, '', text)

  inf_dic4[key] = text

import urllib
import urllib.parse
import urllib.request
url = 'https://www.mediawiki.org/w/api.php?action=query&titles=File:' + urllib.parse.quote(inf_dic4['国旗画像']) + '&format=json&prop=imageinfo&iiprop=url'
connection = urllib.request.urlopen(urllib.request.Request(url))
response = json.loads(connection.read().decode())
print(response['query']['pages']['-1']['imageinfo'][0]['url'])

#以下でも似たことができる

# import pandas as pd
# import re
# import requests
# pattern = re.compile('\|(.+?)\s=\s*(.+)')
# wiki = pd.read_json('jawiki-country.json.gz', lines = True)
# uk = wiki[wiki['title']=='イギリス'].text.values
# ls = uk[0].split('\n')
# d = {}
# for line in ls:
#     r = re.search(pattern, line)
#     if r:
#         d[r[1]]=r[2]
        
# S = requests.Session()
# URL = "https://commons.wikimedia.org/w/api.php"
# PARAMS = {
#     "action": "query",
#     "format": "json",
#     "titles": "File:" + d['国旗画像'],
#     "prop": "imageinfo",
#     "iiprop":"url"
# }
# R = S.get(url=URL, params=PARAMS)
# DATA = R.json()
# PAGES = DATA['query']['pages']
# for k, v in PAGES.items():
#     print (v['imageinfo'][0]['url'])
