sent_list = 'Now I need a drink, alchoholic of course, after the heavy lectures involving quantum mechanics'
letter_count = 0
letter_count_list = []
for i in range(len(sent_list)):
    if sent_list[i] == ' ':
        letter_count_list.append(letter_count)
        letter_count = 0
        
    else:
        letter_count += 1

print(letter_count_list)