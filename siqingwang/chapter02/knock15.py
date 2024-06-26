# 15. Last N lines

N = int(input('N = '))

with open(file_path, 'r') as file:
  last_lines = file.readlines()[-N:]

for line in last_lines:
  print(line, end='')