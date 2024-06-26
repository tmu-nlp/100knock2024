# 11. Replace tabs into spaces

file_path = '/Users/hoshikawakiyoru/Library/Mobile Documents/com~apple~CloudDocs/ソーシャル・データサイエンス/NLP/popular-names.txt'

with open(file_path, 'r') as file:
  content = file.read()

content = content.replace('\t', '')