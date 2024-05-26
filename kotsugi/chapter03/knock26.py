import gzip
import json
import re

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

s1 = r'\'{2,5}(.*?)\'{2,5}'

pattern = r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))'
fields = re.findall(pattern, re.sub(s1, '\\1', matches[0]), re.DOTALL + re.MULTILINE)

dct = {}

for f in fields:
  dct[f[0]] = f[1]

for d in dct.keys():
  print(f'{d}:\t{dct[d]}')
