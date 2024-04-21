sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

# 単語に分割
words = sentence.replace(".", "").split()

# 先頭1文字を取る単語の位置（1始まりで指定）
one_letter_positions = {1, 5, 6, 7, 8, 9, 15, 16, 19}

# 元素記号の辞書を作成
elements = {}

for index, word in enumerate(words, start=1):
    if index in one_letter_positions:
        # 指定された位置の単語は1文字を取る
        element_symbol = word[0]
    else:
        # それ以外の単語は2文字を取る
        element_symbol = word[:2]
    elements[element_symbol] = index

print(elements)