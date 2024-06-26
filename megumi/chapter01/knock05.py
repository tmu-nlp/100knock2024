#05.n_gram
#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

text05="I am an NLPer"

def ngram(n, word):
    list=[]
    for i in range(len(word) - n + 1):#len(word)-n+1回、文字列を１文字ずつずらしながらリストに追加
        list.append(word[i:i+n])
    return list

print(f"単語bi-gram:{ngram(2,text05.split()}"))
print(f"文字bi-gram:{ngram(2,text05)}")      
      