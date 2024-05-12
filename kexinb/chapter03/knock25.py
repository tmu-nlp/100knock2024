# task25. テンプレートの抽出
# 記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．

import pandas as pd
import re
from knock20 import load_uk_data

uk_data = load_uk_data('chapter03/jawiki-country.json.gz')

result = {}
template = re.findall(r"^\{\{基礎情報.*?$(.*?)^(\}\}$)", uk_data, 
                      re.MULTILINE + re.DOTALL)

for line in template[0][0].split("\n"):
    if re.search(r"^\|.+?\s*=\s*",line):
        field = (re.findall(r"^\|(.+?)\s*=\s*(.+)", line))
        result[field[0][0]] = field[0][1]

if __name__ == "__main__":
    for key, value in result.items():
        print(key, "\t", value)