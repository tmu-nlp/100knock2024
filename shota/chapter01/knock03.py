text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
print(text.split())

for i in text.split():
    length = len(i)
    if "," in i:
        length -= 1
    if "." in i:
        length -= 1
    print(length,end = " ")