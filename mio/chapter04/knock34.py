#34. 名詞の連接
#名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

#用語　最長一致：条件を満たすものが複数あった場合、長いほうを選ぶ

import knock30


conj_nouns = set()
conj_noun = ""
counter = 0

#方針：ある形態素の品詞が名詞のとき、その表層形をconj_nounに追加→カウンタ変数を更新
#　　　　　　　　　　　ではなく、カウンタ変数が1より大きいとき、（すでにconj_nounに格納済みの名詞の連接を）をconj_nounsに格納
#　　　カウンタ変数を０に/名詞の連接一組を格納しておく文字列を空に、それぞれリセット

for sentence in knock30.morph_results:
  #作業１
  for i in range(len(sentence)):
    if sentence[i]["pos"] == "名詞":
      conj_noun += sentence[i]["surface"]
      counter += 1
    else:
        if counter > 1:
            conj_nouns.add(conj_noun)
        counter = 0
        conj_noun = ""
        
print(conj_nouns)