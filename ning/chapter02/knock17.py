with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    names = sorted(set(x.split(' ')[0] for x in lines))
print(names)
