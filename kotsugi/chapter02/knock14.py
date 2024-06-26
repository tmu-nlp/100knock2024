import sys

args = sys.argv
count = 0

n = int(args[1])

with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  for line in f:
    count += 1
    if (count <= n):
      print(line.strip())
    else:
      break

# head -n 10 ./woodnx/chapter02/popular-names.txt
