#31.動詞
#動詞の表層形をすべて抽出せよ．
 
import knock30
result = knock30.parse_neko()

#空のsetを作成する。
se = set()

#問30で作成したresultを反復処理させる。
for lis in result:
  for dic in lis:
    #品詞が動詞ならば、表層形をsetに格納していく。
    if dic["pos"] == "動詞":
      se.add(dic["surface"])

print(se)

"""
出力結果
{'解せ', '抜こ', 'やい', 'ふくれ', 
'察せ', '生かし', 'ときゃ', '気に入ら', 
'引き立た', '磨る', '横切っ', 'さまし', 
'洩らし', 'ゆるん', '怒っ', '死に', 
'いらっしゃれ', '誘い出す', '凌い'
"""