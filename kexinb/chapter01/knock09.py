# task09: Typoglycemia
# スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．ただし，長さが４以下の単語は並び替えないこととする．適当な英語の文（例えば”I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .”）を与え，その実行結果を確認せよ．

import random
import string

def shuffle_word(word): #string
    if len(word) < 4:
        return word
    else:
        return "".join([word[0]] + random.sample(word[1:-1],len(word)-2) + [word[-1]])

def shuffle_sentence(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    wordList = text.split()
    return " ".join([shuffle_word(w) for w in wordList])

print(shuffle_sentence("NLP is fun!")) 
print(shuffle_sentence("NLP is super dope!")) 
print(shuffle_sentence("I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .")) 