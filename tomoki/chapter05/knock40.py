#No40(係り受け解析結果の読み込み（形態素))
#メンバ変数はクラスの中にある変数のこと
#クラス：設計図
#インスタンス：実際に作った物
#オブジェクト：モノ（クラスとかインスタンスとかをふんわりと表現したもの）
class Morph:
  #コンストラクタを定義する
  def __init__(self, line):
    #lineをsurfaceとotherに分離する
    surface, other = line.split("\t")
    #otherを,で分けていく
    other = other.split(",")
    #インスタンス変数に値を受け取らせる
    #表層形、基本形、品詞、品詞細分類１
    self.surface = surface
    self.base = other[-3]
    self.pos = other[0]
    self.pos1 = other[1]
 
sentences = [] #文リスト
morphs = [] #形態素リスト
 
with open("ai.ja.txt.parsed") as f:
  #ファイルを一行ずつ読み込む
  for line in f:
    #アスタリスク(文節の開始位置)なら処理を飛ばす
    if line[0] == "*":
      continue
    #EOSでないならクラスを呼び出す(selfはインスタンス自身を表すから、引数はlineのみ)
    elif line != "EOS\n": 
      morphs.append(Morph(line))
    else:  #EOS（文末）の場合、sentencesリストにいれ,morphsリストをリセットする。
      sentences.append(morphs)
      morphs = []
 
for i in sentences[0]:
    print(vars(i))