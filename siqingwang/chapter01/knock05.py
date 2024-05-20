# 05. n-gram
def extract_ngrams(sequence, n):
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = sequence[i:i+n]
        ngrams.append(ngram)
    return ngrams

# Example usage:
sentence = "I am an NLPer"

# Word bi-grams
word_bigrams = extract_ngrams(sentence.split(), 2)
print("Word bi-grams:", word_bigrams)

# Letter bi-grams
letter_bigrams = extract_ngrams(sentence, 2)
print("Letter bi-grams:", letter_bigrams)
