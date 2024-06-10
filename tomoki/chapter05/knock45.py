#NO45(動詞の格パターンの抽出)
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
#result45.txtを作成する。
with open("./result45.txt", "w") as f:
  #動詞を含む文節を探し出す。
  for i in range(len(sentences)):
    for chunk in sentences[i].chunks:
      for morph in chunk.morphs:
        # 動詞を含む文節から助詞を取り出す。
        if morph.pos == "動詞": 
          particles = []
          for src in chunk.srcs:
            particles += [morph.surface for morph in sentences[i].chunks[src].morphs if morph.pos == "助詞"]
        #取り出した助詞が2個以上の場合、集合にし、辞書順に並び替え、１つスペースをあけて結合する。
          if len(particles) > 1:
            particles = set(particles)
            particles = sorted(list(particles))
            form = " ".join(particles)
            #file=fを忘れずに(これがないとresult45.txtが何も書かれていない状態になる)
            print(f"{morph.base}\t{form}", file=f)

        #UNIXコマンド   
        #cat ./result45.txt | sort | uniq -c | sort -nr 
        # cat ファイルを表示する
        # -n データを文字ではなく、数値として並び変える 
        # -r 降順に並び替える
        # -c 重複行を取り除き、それぞれの行が何回出現したかを数える