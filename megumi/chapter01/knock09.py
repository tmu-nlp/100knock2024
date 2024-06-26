#09.Typoglycemia
#スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
#ただし，長さが４以下の単語は並び替えないこととする．
#適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，その実行結果を確認せよ．

import random

text09="I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

list=[]
for i in text09.split():
    if len (i)>4:
        i=i[0]+"".join(random.sample(i[1:-1], len(i)-2))+i[-1]
    list.append(i)

print(" ".join(list))