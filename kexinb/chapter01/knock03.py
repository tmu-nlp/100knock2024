# task03: 円周率
# “Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”\
# という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

import string

def strip_puncs(raw):
    ### translate(translation table) 
    ### maketrans(chars to be replaced, their replacements, chars to be deleted) -> translation table
    return raw.translate(str.maketrans('', '', string.punctuation))

def words_len(raw):
    text = strip_puncs(raw)
    return [len(word) for word in text.split()]

if __name__ == "__main__":
    rawText = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    print(words_len(rawText))