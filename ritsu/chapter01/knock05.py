# n-gram...「与えられたシーケンスからn個の連続する要素を抽出する手法」

def n_gram(sequence, n):
    """
    sequemce: 入力するシーケンス
    n: 抽出する数を指定
    """
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)] 

sentence = "I am an NLPer"

# 単語bi-gram
words = sentence.split()
word_bi_gram = n_gram(words, 2)

# 文字bi-gram
char_bi_gram = n_gram(sentence.replace(" ", ""), 2)  # スペースを除去してからbi-gramを生成

print("単語bi-gram:", word_bi_gram)
print("文字bi-gram:", char_bi_gram)