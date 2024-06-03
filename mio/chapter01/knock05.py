#05. n-gramPermalink
#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
#この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．
#先頭を１文字ずつずらしながらリストに追加
#先頭になれるのは左からlen(word)-n+1番目の単語or文字のみ（右からn-1番目の単語or文字は先頭になれない）

def n_gram(n, word):
  list = []
  for i in range(len(word)-n+1):
    list.append(word[i:i+n])
  return list
n=2
sequence = "I am an NLPer"
print(f"文字bi-gram：{n_gram(n, sequence)}")
print(f"単語bi-gram：{n_gram(n, sequence.split())}")