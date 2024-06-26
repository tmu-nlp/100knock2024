def n_gram(n, str05):
    letter = str05.replace(" ", "")
    word = str05.split(" ")
    letter_list = []
    word_list = []
    for i in range(len(letter)):
        letter_list.append(letter[i:n+i])
    for j in range(len(word)):
        word_list.append(word[j:n+j])
    return letter_list, word_list

str6a = "paraparaparadise"
str6b = "paragraph"
X_list, _ = n_gram(2, str6a)
Y_list, _ = n_gram(2, str6b)
X = set(X_list)
Y = set(Y_list)
print("和集合: ", X | Y)
print("積集合: ", X & Y)
print("差集合: ", X - Y)

word = "se"
if word in X and word in Y:
  print("XとYの積集合が{}含まれる".format(word))
elif word in X:
  print("Xにのみ{}が含まれる".format(word))
elif word in Y:
  print("Yにのみ{}が含まれる".format(word))
else:
  print("XとYには{}が含まれない".format(word))