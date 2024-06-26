import sys

args = sys.argv
count = 0

n = int(args[1])

with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  lines = f.readlines()[-n:]
  for line in lines:
    count += 1
    if (count <= n):
      print(line.strip())
    else:
      break

# tail -n 10 ./woodnx/chapter02/popular-names.txt
