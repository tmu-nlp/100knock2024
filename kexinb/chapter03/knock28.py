# task28. MediaWikiマークアップの除去
# 27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，国の基本情報を整形せよ

import re
from knock25 import template

result = {}
for line in template[0][0].split("\n"):
    if re.search(r"^\|.+?\s=\s*",line):
        field = re.findall(r"^\|(.+?)\s*=\s*(.+)",line)
        remove_lang = re.sub(r"\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}",r"\1",field[0][1])
        remove_markup = re.sub(r"\[\[(?:[^|]+?\|)?(.+?)\]\]",r"\1",remove_lang)
        remove_http = re.sub(r"\[http.+?\]",r"",remove_markup)# http/https
        remove_ref = re.sub(r"<.+?>",r"",remove_http) # </ref>
        remove_cite = re.sub(r"\{\{(.*?)\}\}",r"",remove_ref)
        result[field[0][0]] = remove_cite

if __name__ == "__main__":
    for key, value in result.items():
        print(key,"\t",value)