def n_gram_devider(sequence, n):
    words = sequence.split()
    word_n_gram = []
    letter_n_gram = []
    for i in range(len(words)-n+1):
        word_n_gram.append(words[i:i+n])
    for i in range(len(sequence)-n+1):
        letter_n_gram.append(sequence[i:i+n])

    return word_n_gram, letter_n_gram

seq = 'I am an NLPer'
word_ngr, let_ngr = n_gram_devider(seq,2)
print(word_ngr,let_ngr)