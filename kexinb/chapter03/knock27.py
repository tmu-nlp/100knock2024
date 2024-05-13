# task27. 内部リンクの除去
# 26の処理に加えて，テンプレートの値からMediaWikiの内部リンクマークアップを除去し，テキストに変換せよ

import re
from knock25 import template

result = {}
for line in template[0][0].split("\n"):
    if re.search(r"^\|.+?\s=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        remove_quote = re.sub(r"\'{2,5}",r"",field[0][1]) 
        remove_markup = re.sub(r"\[\[(?:[^|]+?\|)?(.+?)\]\]",r"\1",remove_quote)
        result[field[0][0]] = remove_markup

for key, value in result.items():
    print(key,"\t",value)