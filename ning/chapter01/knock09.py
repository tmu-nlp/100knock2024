import random

def swords(text):
    words = text.split()
    s_words = []

    for word in words:
        if len(word) <= 4:
            s_words.append(word)#長さが４以下の単語は並び替えないこと
        else:
            first_char = word[0]
            last_char = word[-1]
            middle_chars = list(word[1:-1])
            random.shuffle(middle_chars)
            s_word = ''.join([first_char] + middle_chars + [last_char])
            s_words.append(s_word)

    return ' '.join(s_words)

text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

s_sentence = swords(text)
print(s_sentence)
