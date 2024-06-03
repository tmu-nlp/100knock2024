#No42(係り元と係り先の文節の表示)
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
  #表示結果をテキスト化する
  with open("./result42.txt2", "w") as f2:
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
        #文節の係り先番号を取得(ex: D42→42)
         dst = int(line.split()[2].replace("D", ""))
       #文節の開始位置でも文末でもない場合、Morphクラスを呼び出し、それをmorphsリストに追加
       elif line != "EOS\n":
         morphs.append(Morph(line))
       #上記の2つの条件に当てはまらない場合、Chunkクラスを呼び出し、それをchunksリストに追加
       #その後、chunksを引数としてSentenceクラスを呼び出し、それをsentencesリストに追加し、それぞれのリストをリセットする
       else:
         chunks.append(Chunk(morphs, dst, chunk_id))
         sentences.append(Sentence(chunks))
         morphs = []
         chunks = []
         dst = None
         chunk_id = 0
     #sentences[2].chunksをchunkにいれていく(係り先番号を取得するために利用する)
     for chunk in sentences[2].chunks:
       #dstが-1の場合、処理を飛ばす(係り先文節がないということ)
       if chunk.dst == -1:
         continue
       #係り先文節がある場合
       else:
         #文節内のテキストを連結させる
         #もしmorph.posが記号でないなら、リスト内包表記を使って、morph.surface(表層形)をリストにして、joinメソッドで空文字において連結する）
         surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"])
         #もしmorph.posが記号でないなら、係り先文節内のテキストを連結させる
         next_surf = "".join([morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos != "記号"]) 
         #result42.txt"にprint結果を入れる
         print(f"{surf}\t{next_surf}",file=f2)
