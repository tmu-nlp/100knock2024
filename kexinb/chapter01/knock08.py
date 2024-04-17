# task08: 暗号文
# 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
#     英小文字ならば(219 - 文字コード)の文字に置換
#     その他の文字はそのまま出力
#     この関数を用い，英語のメッセージを暗号化・復号化せよ．

def cipher(text):
    text = [chr(219 - ord(word)) if 97 <= ord(word) <= 122 else word for word in text]
    return ''.join(text)

test1 = "Aaaaaaa"
print(cipher(test1))
test2 = "abcde"
print(cipher(test2))
test3 = "ABCDE"
print(cipher(test3))
test4 = "NLP is fun!"
print(cipher(test4))