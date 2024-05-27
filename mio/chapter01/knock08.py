#08. 暗号文
#08 与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
#英小文字ならば(219 - 文字コード)の文字に置換
#その他の文字はそのまま出力
#この関数を用い，英語のメッセージを暗号化・復号化せよ．
#参考：https://qiita.com/segavvy/items/5552623de614ca3344df

def cipher(string):
  result = ""
  for letter in string:
    if letter.islower() == True:
      result += chr(219 -ord(letter))
    else:
      result += letter
  return result
#暗号化
string = input("メッセージを入力してください：")
encryption = cipher(string)
print(encryption)
#復号化
decryption = cipher(encryption)
print(decryption)