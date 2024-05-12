# task29. 国旗画像のURLを取得する
# テンプレートの内容を利用し，国旗画像のURLを取得せよ．
#（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

import requests
from knock28 import result

s = requests.Session()

params = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:" + result["国旗画像"],
    "iiprop": "url",
}

r = s.get("https://en.wikipedia.org/w/api.php", params=params)
data = r.json()
pages = data["query"]["pages"]

for k, v in pages.items():
    print(v['imageinfo'][0]['url'])
