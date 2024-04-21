elemental_symbols = ' Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
num_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]

elm_dict = {}
word_count = 0
i = 0

while i < len(elemental_symbols):
    if elemental_symbols[i] == ' ':
        word_count += 1
    if word_count in num_list and elemental_symbols[i-1]== ' ':
        key = elemental_symbols[i] 
        elm_dict[key] = word_count
        i += 1
    elif elemental_symbols[i-1]== ' ':
        key = elemental_symbols[i] + elemental_symbols[i+1]
        elm_dict[key] = word_count
    i += 1

print(elm_dict)