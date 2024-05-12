def n_gram(s, n):
    ngrams = set()
    for i in range(len(s) - n + 1):
        ngrams.add(s[i:i+n])
    return ngrams

# 文字bi-gramの集合を生成
word1 = "paraparaparadise"
word2 = "paragraph"
X = n_gram(word1, 2)
Y = n_gram(word2, 2)

# 和集合
u_set = X | Y

# 積集合
i_set = X & Y

# 差集合
d_set = X - Y

# ’se’というbi-gramがXおよびYに含まれるかどうかを調べる
se_in_X = 'se' in X
se_in_Y = 'se' in Y

print("X:", X)
print("Y:", Y)
print("和集合:", u_set)
print("積集合:", i_set)
print("差集合(X - Y):", d_set)
print("'se' in X:", se_in_X)
print("'se' in Y:", se_in_Y)
