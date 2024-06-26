word_1 = "パトカー"
word_2 = "タクシー"

# 空の変数を用意
connected_word = ""
# 2つのワードで文字を繰り返し取得する
for i in range(4):
    connected_word += word_1[i] + word_2[i]
print(connected_word)