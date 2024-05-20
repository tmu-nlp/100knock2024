str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

splited_list = str.replace(',', '').replace('.', '').split()

for i in splited_list:
  print(len(i), end="")

print() # 改行のためのprint
