sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

# 句読点を除去し、単語に分割
words = sentence.replace(',', '').replace('.', '').split()

# 各単語の文字数をカウント
word_lengths = [len(word) for word in words]

print(word_lengths)