#NO46(動詞の格フレーム情報の抽出)
class Sentence:
    #コンストラクタを定義する
  def __init__(self, chunks):
    #インスタンス変数に値を受け取らせる
    self.chunks = chunks
    #self.chunksの中身とインデックス番号をchunkとiに入れていく
    for i, chunk in enumerate(self.chunks):
      #chunk.dstにNoneも-1も含まれない場合、srcsリストにインデックス番号を追加する
      if chunk.dst not in [None, -1]:
        self.chunks[chunk.dst].srcs.append(i)
 
class Chunk:
  #コンストラクタを定義する
  def __init__(self, morphs, dst, chunk_id):
    #インスタンス変数に値を受け取らせる
    self.morphs = morphs
    self.dst = dst
    self.srcs = []
    self.chunk_id = chunk_id

class Morph:
  #コンストラクタを定義する
  def __init__(self, line):
    #lineをsurfaceとotherに分離する
    surface, other = line.split("\t")
    #otherを,で分けていく
    other = other.split(",")
    #インスタンス変数に値を受け取らせる
    self.surface = surface
    self.base = other[-3]
    self.pos = other[0]
    self.pos1 = other[1]
 
sentences = [] #文リスト
chunks = [] #節リスト
morphs = [] #形態素リスト
chunk_id = 0 #文節番号
 
with open("ai.ja.txt.parsed") as f:
  #ファイルを一行ずつ読み込む
  for line in f:
    #アスタリスクは文節の開始位置を意味する
    if line[0] == "*":
      #morphsリストに何か入っている場合、Chunkクラスを呼び出し、それをchunksリストに追加
      #その後、chunk_idを1増やし、morphsリストをリセットする
      if morphs:
        chunks.append(Chunk(morphs, dst, chunk_id))
        chunk_id += 1
        morphs = []
      dst = int(line.split()[2].replace("D", ""))
    #文節の開始位置でも文末でもない場合、Morohクラスを呼び出し、それをmorphsリストに追加
    elif line != "EOS\n":
      morphs.append(Morph(line))
    #上記の2つの条件に当てはまらない場合、Chunkクラスを呼び出し、それをchunksリストに追加
    #その後、chanksを引数としてSentenceクラスを呼び出し、それをsentencesリストに追加し、それぞれのリストをリセットする
    else:
      chunks.append(Chunk(morphs, dst, chunk_id))
      sentences.append(Sentence(chunks))
      morphs = []
      chunks = []
      dst = None
      chunk_id = 0
#result46.txtを作成する
with open("./result46.txt", "w") as f:
  #最左動詞を見つける
  for i in range(len(sentences)):
    for chunk in sentences[i].chunks:
      for morph in chunk.morphs:
        if morph.pos == "動詞": 
          particles = []
          items = []
          #係り元のインデックス番号をfor文で回す
          for src in chunk.srcs:
          #係り受けの文節を調べていく
          #もし助詞が見つかったら、その表層形を「particles」に入れる（リスト内包表記）
            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"]
          #もし記号以外であれば、その表層形を結合する。
            items += ["".join([morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos != "記号"])]
          #particlesとitemsが2以上(係っているものが2つ以上の時)
          if len(particles) > 1:
            if len(items) > 1:
              #partclesとitemsの集合を辞書順にする
              particles = sorted(set(particles))
              items = sorted(set(items))
              #partclesとitemsとpredicateをそれぞれ空白をあけて結合する
              particles_form = " ".join(particles)
              items_form = " ".join(items)
              #上記の３つをタブ区切りで出力＆fileに記述する
              print(f"{morph.base}\t{particles_form}\t{items_form}", file=f)
