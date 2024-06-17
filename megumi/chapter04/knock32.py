#32.動詞の基本形
#動詞の基本形をすべて抽出せよ

import knock30
result=knock30.parse_neko()

se = set()

for lis in result:
  for dic in lis:
    if dic["pos"] == "動詞":
      se.add(dic["base"])

print(se)

"""
出力結果
{'載せる', '供する', '試みる', 
'引きあげる', 'くばる', '取り違える', 
'漬ける', '振れる', '割る', '観る', 
'砕ける', '見くびる', '払う', '知る', 
'引き取る', 'あばれる', '感ずる', 
'つとめる', '褒める', '喰う
"""