#機能動詞構文のマイニング
"""
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

・「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
・述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
・述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
・述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
例文　　「太郎は食事をする。」
構造　　「食事を」 (サ変接続名詞 + を) 
　　　　「する」 (動詞)
出力　　　食事をする	を	太郎
"""

#41で定義した関数を用いる
import knock41
      
with open("./result47.txt", "w") as f:
  for sentence in knock41.sentences:
    for chunk in sentence.chunks: #文から文節を取り出す
      for morph in chunk.morphs:  #文節から単語を取り出す
        if morph.pos == "動詞":    #単語が動詞の場合動詞に係っている文節を調べる
          for src in chunk.srcs:  
            predicates = []       #「サ変接続名詞+を＋動詞の基本形」を保存
            #文節の長さが２　＃最初の形態素の品詞細分類が「サ変接続」　＃２番目の形態素の表層形が「を」
            if len(sentence.chunks[src].morphs) == 2 and sentence.chunks[src].morphs[0].pos1 == "サ変接続" and sentence.chunks[src].morphs[1].surface == "を":
              #連結して述語を作成
              predicates = "".join([sentence.chunks[src].morphs[0].surface, sentence.chunks[src].morphs[1].surface, morph.base])
              particles = [] #再度、文節内に動詞が出現したとき、保存していた助詞をリセット
              items = []     #再度、文節内に動詞が出現したとき、保存していた項をリセット
              for src in chunk.srcs: #文内に、items変数を追加し、係り元文節の単語列（項）を取得する。
                particles += [morph.surface for morph in sentence.chunks[src].morphs if morph.pos == "助詞"]
                             #文節内の次の動詞が出現するまで、動詞に係る助詞を保存しておく。
                item = "".join([morph.surface for morph in sentence.chunks[src].morphs if morph.pos != "記号"])
                item = item.rstrip()
                if item not in predicates:
                  items.append(item)
              #助詞と項が複数ある場合
              if len(particles) > 1:
                if len(items) > 1:
                  #辞書順に並べ替える
                  particles = sorted(set(particles))
                  items = sorted(set(items))
                  particles_form = " ".join(particles)
                  items_form = " ".join(items)
                  predicate = " ".join(predicates)
                  #述語と述語に係っている助詞、項をタブ区切りでファイルに書き込む。
                  print(f"{predicates}\t{particles_form}\t{items_form}", file=f)