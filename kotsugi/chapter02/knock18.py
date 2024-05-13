lst = []

with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  filelines = f.readlines()

  for i, line in enumerate(filelines):
    item = line.strip().split('\t')
    dct = {
      'first': item[0],
      'second': item[1],
      'third': item[2],
      'fourth': item[3],
    }
    lst.append(dct)

  sorted_lst = sorted(lst, key=lambda x: x['third'], reverse=True)
  # ラムダ式（無名関数）を用いて，ソート対象のキーを指定する

  for line in sorted_lst:
    print(f"{line['first']}\t{line['second']}\t{line['third']}\t{line['fourth']}\t")

# sort -k 3,3 -r woodnx/chapter02/popular-names.txt 
