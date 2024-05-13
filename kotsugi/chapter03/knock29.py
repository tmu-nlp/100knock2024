import gzip
import json
import re
import requests

def multi_sub(patterns, text):
  for pattern in patterns:
    text = re.sub(pattern[0], pattern[1], text)
  return text

data = {}

with gzip.open('./woodnx/chapter03/jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
  for line in f:
    # JSON文字列をPython辞書に変換してリストに追加
    data = json.loads(line)
    if (data["title"] == "イギリス"):
      break

content = data["text"]
pattern = r'\{\{基礎情報.*?\n(.*?)\n\}\}$'
matches = re.findall(pattern, content, re.DOTALL + re.MULTILINE)
# re.DOTALL: 「.」に対して改行を含めるオプション(from: https://qiita.com/FukuharaYohei/items/459f27f0d7bbba551af7#dotall)
# re.DOTALL: 複数行に対して検索するオプション(from: https://qiita.com/FukuharaYohei/items/459f27f0d7bbba551af7#multiline)

s = [
  [ r'\'{2,5}(.*?)\'{2,5}', '\\1' ],
  [ r'\[\[(.*?)(?:|(\|.*?)|(\#.*?\|.*?))\]\]', '\\1' ],
  [ r'<ref.*>.*?(?:</ref>|)', '' ],
  [ r'<br />', '' ],
  [ r'\{\{0\}\}', '' ],
  [ r':en:(.*?)', '\\1'],
  [ r'\{\{仮リンク\|(.*?)\|en\|(.*?)\}\}', '\\1'],
  [ r'</ref>', ''],
  [ r'\**{{lang\|.*?\|(.*?)}}', '\\1'],
  [ r'(.*?){{en icon}}(.*?)\{\{center\|ファイル:.*?\}\}', '\\1 \\ \\2'],
]

pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
fields = re.findall(
  pattern, 
  multi_sub(s, matches[0]), 
  re.DOTALL + re.MULTILINE
)

dct = {}

for f in fields:
  dct[f[0]] = f[1]

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
  "action": "query",
  "format": "json",
  "prop": "imageinfo",
  "titles": f'File:{dct["国旗画像"]}',
  "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]

for k, v in PAGES.items():
  print(f"flug's url: {v["imageinfo"][0]["url"]}")
