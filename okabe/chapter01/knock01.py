str_list = 'パタトクカシーー'
return_list = ''
for i in range(len(str_list)):
    if (i+1) % 2 == 0:
        return_list += str_list[i]
print(return_list)