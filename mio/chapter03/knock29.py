#問29
#テンプレートの内容を利用し，国旗画像のURLを取得せよ．（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）
import pandas as pd
import re 
import requests

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]
dic = {}

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{dic['国旗画像']}",
    "iiprop":"url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA["query"]["pages"]

for k, v in PAGES.items():
    print(v["imageinfo"][0]["url"])