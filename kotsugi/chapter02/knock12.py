col1 = []
col2 = []
with open("./woodnx/chapter02/popular-names.txt", "r") as rf:
  for line in rf:
    cols = line.split('\t')
    col1.append(cols[0])
    col2.append(cols[1])

with open("./woodnx/chapter02/out/col1.txt", "w+") as wf:
  wf.write("\n".join(col1))


with open("./woodnx/chapter02/out/col2.txt", "w+") as wf:
  wf.write("\n".join(col2))

# cut -f 1 "./woodnx/chapter02/popular-names.txt" | less
# cut -f 2 "./woodnx/chapter02/popular-names.txt" | less
