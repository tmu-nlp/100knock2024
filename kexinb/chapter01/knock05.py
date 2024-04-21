# task05: n-gram
# 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

def n_gram(seq, n):
    return [seq[i:i+n] for i in range(len(seq)-n+1)]

if __name__ == "__main__":
    text = "I am an NLPer"
    print(n_gram(text.split(), 2))