import collections

with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    temp = [line.split() for line in lines]
    c = collections.Counter(name[0] for name in temp)
    c2 = c.most_common()

print(c2)

#cut -d ' ' -f 1 "[PATH]/popular-names.txt"|sort|uniq -c|sort -r -n