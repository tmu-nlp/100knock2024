import random
text09 = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind."
text09 = text09.replace(".", "")
text09 = text09.split(" ")
temp = ""
for i in text09:
    if len(i) <= 4:
        pass
    else:
        i = i[0]+"".join(random.sample(i[1:-1], len(i[1:-1])))+i[-1]
    temp += i + " "
result = temp[:-1] + "."
print(result)
