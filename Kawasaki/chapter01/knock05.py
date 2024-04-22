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

bi_gram_letter, bi_gram_word = n_gram(2, "I am an NLPer")
print(bi_gram_letter)
print(bi_gram_word)