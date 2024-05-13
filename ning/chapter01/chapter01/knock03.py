text = '''Now I need a drink, alcoholic of course, 
after the heavy lectures involving quantum mechanics.'''
n_list = []
temp = ""

text = text.replace(',', '')
text = text.replace('.', '')#句読点の処理

for i in text:
    if i == " " :#単語長さ？の検出
        n_list.append(len(temp))#長さを文字列に追加
        temp = ""
    else:
        temp += i

print(n_list)

