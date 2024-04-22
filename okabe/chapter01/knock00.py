str_list = 'stressed'
reversed = ''
for i in range(len(str_list)):
    reversed += str_list[-(i+1)]
print(reversed)