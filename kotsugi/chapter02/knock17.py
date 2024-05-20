alphabets = set()

with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  for line in f:
    first = line.split('\t')[0]
    alphabets.add(first)

print(sorted(alphabets))

# cut -f 1 ./woodnx/chapter02/popular-names.txt | sort | uniq
