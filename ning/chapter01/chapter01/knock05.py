def n_gram(sequence, n):
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngrams.append(sequence[i:i+n])
    return ngrams

#uni-gram(N=1)
#bi-gram(N=2)：
#tri-gram(N=3)

text = "I am an NLPer"
w_bigram = n_gram(text.split(), 2)

ch_bigram = n_gram(text, 2)

print("単語bi-gram:", w_bigram)
print("文字bi-gram:", ch_bigram)
