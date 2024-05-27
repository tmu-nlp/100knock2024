dct = {}
with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  for line in f:
    name = line.split('\t')[0]
    if (name in dct):
      dct[name] += 1
    else:
      dct[name] = 1

sorted_dct = sorted(dct.items(), key=lambda x:x[1], reverse=True)

for item in sorted_dct:
  print(f'{item[0]}: {item[1]}')

# cut -f 1 ./woodnx/chapter02/popular-names.txt | sort | uniq -c | sort -r -n -k 1,1