import re
#.は文字*は繰り返すこと?は最小単位で区切る
pattern = "\[\[Category:.*?\]\]"
result = re.findall(pattern, UK_text)
result