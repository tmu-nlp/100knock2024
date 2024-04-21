def cipher(text):
   """
   文字列の暗号化、復号化
   以下の仕様で引数の文字列を変換したものを返す
    ・英小文字ならば(219 - 文字コード)の文字に置換
    ・その他の文字はそのまま出力

   変数：
   text:変換する文字列
   result:変換した文字列
   letter:仕様を通じて変換した文字
   """

   result = ''
   for i in text:
      if i.islower():
         letter = chr(219 - ord(i))
         result = result + letter
      else:
         result = result + i
   return result

encode = cipher('I am a NLPer.')
print('暗号化したものは：', encode)
decode = cipher(encode)
print('復号化したものは：', decode)

#　正規表現をもちいたら
# import re
# def cipher(text):
#    """
#    文字列の暗号化、復号化
#    以下の仕様で引数の文字列を変換したものを返す
#     ・英小文字ならば(219 - 文字コード)の文字に置換
#     ・その他の文字はそのまま出力

#    変数：
#    text:変換する文字列
#    result:変換した文字列
#    letter:仕様を通じて変換した文字
#    """
#    repatter = re.compile('[a-z]')
#    result = ''
#    for i in text:
#       if re.search(repatter,i): 
#          letter = chr(219 - ord(i))
#          result = result + letter
#       else:
#          result = result + i
#    return result

# encode = cipher('I am a NLPer.')
# print('暗号化したものは：', encode)
# decode = cipher(encode)
# print('復号化したものは：', decode)