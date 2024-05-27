# task21. カテゴリ名を含む行を抽出
# 記事中でカテゴリ名を宣言している行を抽出せよ．

import re
from knock20 import load_uk_data

uk_data = load_uk_data('chapter03/jawiki-country.json.gz')
uk_data = uk_data.split("\n")

for line in uk_data:
    if re.search(r'^\[\[Category:', line):
        print(line)