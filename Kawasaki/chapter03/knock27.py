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
print(inf_dic3)

#以下でも似たことができる

# import pandas as pd
# import re
# pattern = re.compile('\|(.+?)\s=\s*(.+)')
# p_emp = re.compile('\'{2,}(.+?)\'{2,}')
# p_link = re.compile('\[\[(.+?)\]\]')
# wiki = pd.read_json('jawiki-country.json.gz', lines = True)
# uk = wiki[wiki['title']=='イギリス'].text.values
# lines = uk[0]
# lines = re.sub(p_emp,'\\1', lines)
# lines = re.sub(p_link,'\\1', lines)
# ls = lines.split('\n')
# d = {}
# for line in ls:
#     r = re.search(pattern, line)
#     if r:
#         d[r[1]]=r[2]
# print(d)