str03 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
str03 = str03.replace(",", "")
str03 = str03.replace(".", "")
str03 = str03.split()
list03 = []
for i in str03:
    list03.append(len(i))
print(list03)