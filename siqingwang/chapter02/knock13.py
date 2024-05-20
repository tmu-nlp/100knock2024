# 13. Merging col1.txt and col2.txt

col12_path = '/content/drive/My Drive/NLP/col12.txt'

with open(col1_path, 'r') as col1, open(col2_path, 'r') as col2, open(col12_path, 'w') as col12:
  for line1, line2 in zip(col1, col2):
    col12.write(line1.strip() + ' ' + line2.strip() + '\n')