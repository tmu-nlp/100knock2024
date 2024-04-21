import random

def typoglycemia(sentence):
    """
    与えられたテキスト内の各単語の内部文字の順序をランダムに並び替える。
    各単語の先頭と末尾の文字は固定される。単語の長さが4以下の場合は並び替えない。
    """
    words = sentence.split()
    shuffled_words = [] # 空のリストを用意
    for word in words:
        if len(word) > 4: # 文字数が4以下の場合は処理を行わない
            middle_chars = list(word[1:-1])
            random.shuffle(middle_chars) # randomモジュールのshuffle関数でランダムに並べ替える
            word = word[0] + ''.join(middle_chars) + word[-1] # joinでmiddle_charsを結合
        shuffled_words.append(word)
    return ' '.join(shuffled_words) # word間に空白を入れたいので' 'で結合する

# テストする英語の文
text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

result = typoglycemia(text)
print("Original:", text)
print("Shuffled:", result)