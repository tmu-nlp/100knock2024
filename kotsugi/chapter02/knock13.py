col1 = []
col2 = []

with open("./woodnx/chapter02/out/col1.txt", "r") as f:
  for line in f:
    col1.append(line.strip())

with open("./woodnx/chapter02/out/col2.txt", "r") as f:
  for line in f:
    col2.append(line.strip())

with open("./woodnx/chapter02/out/col3.txt", "w+") as f:
  for i in range(len(min(col1, col2))):
    f.write(f"{col1[i]}\t{col2[i]}\n")

# paste ./woodnx/chapter02/out/col1.txt ./woodnx/chapter02/out/col2.txt 
