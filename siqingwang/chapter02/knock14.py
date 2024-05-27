N = int(input('N = '))  #int:整数

with open(file_path, 'r') as file:
  for i in range(N):
    print(file.readline().strip())