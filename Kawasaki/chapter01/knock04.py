str04 = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
str04 = str04.replace(".", " ")
str04 = str04.split()
num_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]
dict04 = {}
for i,j in enumerate(str04, 1):
    if i in num_list:
        temp = j[0]
        dict04[temp] = i
    else:
        temp = j[:2]
        dict04[temp] = i
print(dict04)