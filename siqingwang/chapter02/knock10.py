# 10. Line count

file_path = '/Users/hoshikawakiyoru/Library/Mobile Documents/com~apple~CloudDocs/ソーシャル・データサイエンス/NLP/popular-names.txt'

with open(file_path, 'r') as file:
  num_lines = sum(1 for line in file)

print(num_lines)