with open("popular-names.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]
    temp = [line.split(' ') for line in lines]
    pnum = sorted(temp, key=lambda x: int(x[2]), reverse=True)

print(pnum)
