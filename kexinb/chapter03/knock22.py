# task22. カテゴリ名の抽出
# 記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

import re
from knock20 import load_uk_data

uk_data = load_uk_data('chapter03/jawiki-country.json.gz')
uk_data = uk_data.split('\n')

for line in uk_data:
    if re.search(r"^\[\[Category:", line):
        print(re.findall(r"\[\[Category:(.*?)(?:\|.*)?\]\]",line))