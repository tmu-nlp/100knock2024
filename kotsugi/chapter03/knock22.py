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
pattern = r'\[\[Category:(.*)\]\]'

matches = re.findall(pattern, content)

for m in matches:
  print(m.replace('|*', ''))