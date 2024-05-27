import json
import re
text_list = [] #空のリストを用意
with open('./jawiki-country.json', 'r') as f: #カレントディレクトリのjawiki-country.jsonファイルをfとして読み取る
    lines = f.readlines() #ファイルの文字列を1行ごとにをリストに入れる
    for line in lines:
        text_list.append(json.loads(line)) #json.loadsは文字列を引数にとり、辞書型にして返す。text_listに追加

for i in range(len(text_list)):
    if text_list[i]['title']=="イギリス": #リストtext_listのi番目のキーがtitleのものについて、値がイギリスかどうか
        UK_text = str(text_list[i]['text']) #記事を抜き出してstr型に変換、UK_textに割り当てる
        break

print(UK_text)

#他の例

# import pandas as pd
# wiki = pd.read_json('jawiki-country.json.gz', lines = True) 
# uk = wiki[wiki['title']=='イギリス']['text'].values[0] #title, textの列があるのでtitleがイギリスの行を条件にしてtextの値を抜き出す

# print(uk)