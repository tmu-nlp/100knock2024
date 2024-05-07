#No29(国旗画像のURLを取得する)
#Pythonからウェブサイトに情報を問い合わせたり、ウェブサイトからデータを収集したりすることができる
#Media Wiki API公式のサンプルコードを参照するとよい
import requests
import pandas as pd
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8 = df[df["title"]=="イギリス"]
D_8= D_8["text"].values
dic={}
for text in D_8[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s*=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
print(dic)
#MediaWiki APIのURL
URL = "https://en.wikipedia.org/w/api.php"
#公式サイトと違うところ　image→imageinfo、　titlesをテンプレートの「国旗画像」の部分を指定するようにする。
#iiprop どのファイル情報を取得するか
PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:{0}".format(dic['国旗画像']),
    "iiprop":"url"}
#上記で設定したurlとparameterを用いて、GETリクエストを送信する。
#ほとんど公式サイトのコードと一緒
R = requests.session().get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA["query"]["pages"]
for k, v in PAGES.items():
    print(v["imageinfo"][0]["url"])