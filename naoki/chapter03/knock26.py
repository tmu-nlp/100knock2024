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
#<>の役割を\で打ち消している
pattern = '基礎情報(.*?\<references/\>)'
result = re.findall(pattern, UK_text)

"""
#?<=Y)XはXの直前にYがある場合にXをマッチする
|国名=イギリス\n|通貨=ポンドというテキストがあった時、国名とイギリス、通貨とポンドというペアを抽出し、それぞれを辞書のキーと値として格納している
(?<=\\\\n\|)：直前に改行とパイプがある部分にマッチします。
(.*?)：項目名をキャプチャします。
*= *：等号とその前後の空白にマッチします。
(.*?)：項目値をキャプチャします。
(?=\\\\n)：直後に改行がある部分にマッチします。
"""
#改行の追加
result[0] += "\\n"
pattern = '(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)'
result2 = re.findall(pattern, result[0])
inf_dic = {}
"""
result2 = [
    ('国名', 'イギリス'),
    ('通貨', 'ポンド'),
    # もっと項目が続く
]
となっているresult2から、
i 項目名
j その値
を取り出し、辞書とする
"""
for i, j in result2:
  inf_dic[i] = j

inf_dic2 = {}
#.items()でキーと値を同時に扱えるようにしている
for key, text in inf_dic.items():
  #テキスト内の//と'が2回以上5回以下連続する部分を空文字列で置換
  if r'(\\\'){2,5}' in text:
    inf_dic[text] = ' '
print(inf_dic)