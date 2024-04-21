text = '''Now I need a drink, alcoholic of course, 
after the heavy lectures involving quantum mechanics.'''
n_list = []
temp = ""

text = text.replace(',', '')
text = text.replace('.', '')

for i in text:
    if i == " " :
        n_list.append(len(temp))
        temp = ""
    else:
        temp += i

print(n_list)

