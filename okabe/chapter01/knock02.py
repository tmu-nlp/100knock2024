str_1 = 'パトカー'
str_2 = 'タクシー'

return_list = ''

for i in range(max(len(str_1),len(str_2))):
    if i+1 <= len(str_1):
        return_list += str_1[i]
    if i+1 <= len(str_2):
        return_list += str_2[i]

print(return_list)