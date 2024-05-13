import re
#<>の役割を\で打ち消している
pattern = '基礎情報(.*?\<references/\>)'
result = re.findall(pattern, UK_text)
#(?<=Y)XはXの直前にYがある場合にXをマッチする
#|国名=イギリス\n|通貨=ポンドというテキストがあった時、国名とイギリス、通貨とポンドというペアを抽出し、それぞれを辞書のキーと値として格納している
#改行の追加
result[0] += "\\n"
pattern = '(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)'
result2 = re.findall(pattern, result[0])
inf_dic = {}
for i, j in result2:
  inf_dic[i] = j
inf_dic