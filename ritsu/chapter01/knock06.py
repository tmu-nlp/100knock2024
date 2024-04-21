def n_gram(sequence, n):
    """
    sequemce: 入力するシーケンス
    n: 抽出する数を指定
    """
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)] 

sentence_X = "paraparaparadise"
sentence_Y = "paragraph"

# set を用いて集合にすることで重複を削除
X = set(n_gram(sentence_X, 2))
Y = set(n_gram(sentence_Y, 2))

# 和集合
union = X | Y

# 積集合
intersection = X & Y

# 差集合 (XからYを引いた集合)
difference = X - Y

# 'se'が各集合に含まれるか
se_in_X = 'se' in X
se_in_Y = 'se' in Y

# 結果を出力
print("X:", X)
print("Y:", Y)
print("和集合:", union)
print("積集合:", intersection)
print("差集合:", difference)
print("'se' in X:", se_in_X)
print("'se' in Y:", se_in_Y)