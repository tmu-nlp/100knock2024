# task23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ.

import re
from knock20 import load_uk_data

uk_data = load_uk_data('chapter03/jawiki-country.json.gz')
uk_data = uk_data.split('\n')

for line in uk_data:
    if re.search(r"^={2,}(.+?)={2,}", line):
        section = re.findall(r"^(={2,})\s*(.+?)\s*(={2,})",line)
        # print(section)
        level = len(section[0][0]) - 1
        print(section[0][1], level)