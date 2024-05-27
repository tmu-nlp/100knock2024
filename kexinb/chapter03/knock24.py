# task24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ．

import pandas as pd
import re
from knock20 import load_uk_data

uk_data = load_uk_data('chapter03/jawiki-country.json.gz')
uk_data = uk_data.split('\n')

for line in uk_data:
    if re.search(r"^\[\[ファイル:(.+?)", line):
        print(re.findall(r"(?:ファイル:)(.+?)\|",line))