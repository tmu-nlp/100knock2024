#06.集合Permmalink
#“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

text6a="paraparaparradise"
text6b="paragragh"

def ngram(n, word):
    list=[]
    for i in range(len(word) - n + 1):#len(word)-n+1回、文字列を１文字ずつずらしながらリストに追加
        list.append(word[i:i+n])
    return list

X = set(ngram(2, text6a))#set()関数(組み込み)：集合オブジェクトに変換。集合オブジェクトに変換することで、集合演算子を使用できるようになる。
Y = set(ngram(2, text6b))

print(f"X:{X}")
print(f"Y:{Y}")

#和集合、積集合、差集合を求める。
print(f"和集合:{X|Y}")
print(f"積集合:{X&Y}")
print(F"差集合:{X-Y}")

#"se"が含まれるかどうか判定
if "se" in X:
    print("seはYに含まれる")
else:
    print("seはYに含まれない")