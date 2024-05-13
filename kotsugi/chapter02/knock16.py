import sys

args = sys.argv
count = 0

n = int(args[1])

with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  lines = f.readlines()
  lines_size = len(lines)
  divides = int(lines_size / n)

  for i in range(n):
    write_lines = lines[i*divides:((i+1)*divides)]
    filename = f'16-{i}.txt'
    with open(f'./woodnx/chapter02/out/{filename}', 'w+') as wf:
      wf.writelines(write_lines)

# split -n 2 woodnx/chapter02/popular-names.txt ./woodnx/chapter02/out/
