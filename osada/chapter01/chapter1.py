
## 問題0　文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ

text1="stressed"
temp=""
for i in reversed(text1):
  temp+= i
print(temp)  

##問題1　「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．
text2= "パタトクカシーー"
temp=""
for i in text2[0:8:2]: 
  temp+=i
print(temp)  

## 問題2　「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ
text3="パトカー"
text31="タクシー"
temp=""

for i3, i31 in zip(text3, text31): #zip関数は複数のリストを同時に取得できる。
  temp+= i3+i31
print(temp)  

##問題3　“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ

text4= "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
text4=text4.replace(",","").replace(".","")
#len関数はオブジェクトのサイズを取得（文字列の長さ、リストの要素数等）
#splitは()を空白にすると空白やタブ、改行等で区切られる。
[len(i) for i in text4.split()] 

##問題4　“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

t5="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
t5=t5.replace(".","")
num=[1, 5, 6, 7, 8, 9, 15, 16, 19]
dic={}

for i,x in enumerate(t5.split()):
  if (i+1) in dic:
    v=x[:1]
  else:
    v=x[:2] 
  dic[v]=i+1
print(dic)  

##5 与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．この関数を用い，”I am an NLPer”という文から単語bi-gram，文字bi-gramを得よ．

#n-gramは連続するｎ個の単語や文字のまとまり.（今日はいい天気ですね　の文字bi-gram　今日　単語bi-gram 今日は）

#defで関数定義 引数を使って何かをする


def N_gram(n,text):
    ngramed=[text[i:i+n] for i in range(len(text))if i+n <= len(text) ] 
    #i:i+2, i=0の時、Iaまでが範囲　range(0,2)は2-1で止まる
    return ngramed

alp="I am an NLPer".replace(" ","") 
word="I am an NLPer".split(" ")

n=2

print(N_gram(n,alp))
print(N_gram(n,word))


##問題6　“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ

t61="paraparaparadise"
t62="paragraph"

X=set(N_gram(2,t61))
Y=set(N_gram(2,t62))

XplusY=X.union(Y)
print(XplusY) #和集合

XandY=X.intersection(Y)
print(XandY)

XminusY=X.difference(Y)
print(XminusY)

print("se" in X)

##7 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def f7(x,y,z):
  return str(x)+"時の"+y+"は"+str(z) 

print(f7(12,"気温",22.4))

##8 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
#英小文字ならば(219 - 文字コード)の文字に置換
#その他の文字はそのまま出力
#この関数を用い，英語のメッセージを暗号化・復号化せよ．

def cipher(t8):
  newt8=""
  for i in t8:
    if i.islower() and i.isalpha():
      i=chr(219-ord(i)) #.islower 小文字 isalpha　アルファベット
    newt8+=i
  return newt8

print(cipher("I am an NLPer"))  

#9 スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．ただし，長さが４以下の単語は並び替えないこととする．適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，その実行結果を確認せよ．


import random

t9="I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
text = t9.split()
result = []
for word in text:
   if len(word) > 4:
       mid = list(word[1:-1]) #1:-1で端以外を取ってこれる
       random.shuffle(mid)
       result.append(word[0] + ''.join(mid) + word[-1])
   else:
       result.append(word)
print(' '.join(result))


  

 

