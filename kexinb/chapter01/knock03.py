# task03: 円周率
# “Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”\
# という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

import string

rawText = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

# strip punctuations --> what's the standard practive?

### translate(translation table) 
### maketrans(chars to be replaced, their replacements, chars to be deleted) -> translation table
text = rawText.translate(str.maketrans('', '', string.punctuation))

print([len(word) for word in text.split()])