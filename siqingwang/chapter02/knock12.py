
file_path = '/content/drive/My Drive/NLP/popular-names.txt'
col1_path = '/content/drive/My Drive/NLP/col1.txt'
col2_path = '/content/drive/My Drive/NLP/col2.txt'

values_list_1 = []
values_list_2 = []

with open(file_path, 'r') as file:
  for line in file:
    cols = line.split()
    if cols:
      values_list_1.append(cols[0])
      values_list_2.append(cols[1])

values_str1 = '\n'.join(values_list_1)
values_str2 = '\n '.join(values_list_2)

with open(col1_path,'w') as col1:
  col1.write(values_str1)

with open(col2_path,'w') as col2:
  col2.write(values_str2)