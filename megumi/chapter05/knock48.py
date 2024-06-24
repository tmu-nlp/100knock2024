#48.名詞から根へのパスの抽出
"""
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
ただし，構文木上のパスは以下の仕様を満たすものとする．
・各文節は（表層形の）形態素列で表現する
・パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

構文木：係り受け解析結果を木構造で表したもの。
"""
class Sentence:
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.dst not in [None, -1]:
                self.chunks[chunk.dst].srcs.append(i)

class Chunk:
    def __init__(self, morphs, dst, chunk_id):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []
        self.chunk_id = chunk_id

class Morph:
    def __init__(self, line):
        surface, other = line.split("\t")
        other = other.split(",")
        self.surface = surface
        self.base = other[6]
        self.pos = other[0]
        self.pos1 = other[1]

sentences = []  # 文リスト
chunks = []     # 節リスト
morphs = []     # 形態素リスト
chunk_id = 0    # 文節番号

with open("./ai.ja.txt.parsed") as f:
    for line in f:
        if line[0] == "*":
            if morphs:
                chunks.append(Chunk(morphs, dst, chunk_id))
                chunk_id += 1
                morphs = []
            dst = int(line.split()[2].replace("D", ""))
        elif line != "EOS\n":
            morphs.append(Morph(line))
        else:
            chunks.append(Chunk(morphs, dst, chunk_id))
            sentences.append(Sentence(chunks))
            morphs = []
            chunks = []
            dst = None
            chunk_id = 0

# 各文の文節で反復させ、文節内に名詞が含まれていれば処理を行う
sentence = sentences[2]
with open("./result48.txt", "w") as f:
    for chunk in sentence.chunks:
        for morph in chunk.morphs:
            if "名詞" in morph.pos:
                # pathで、名詞を含む文節内の単語を連結させる。
                path = ["".join(morph.surface for morph in chunk.morphs if morph.pos != "記号")]
                # 文節の係り先がなくなるまで、係り先文節の形態素列（表層形の）をpathに追加していく。
                while chunk.dst != -1:
                    path.append("".join(morph.surface for morph in sentence.chunks[chunk.dst].morphs if morph.pos != "記号"))
                    chunk = sentence.chunks[chunk.dst]
                # path内の要素を->で連結させて出力する。
                f.write("->".join(path) + "\n")


"""
人工知能->語->研究分野とも->される
される
じんこうちのう->語->研究分野とも->される
される
AI->エーアイとは->語->研究分野とも->される
エーアイとは->語->研究分野とも->される
計算->という->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
概念と->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
コンピュータ->という->道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
道具を->用いて->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
知能を->研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
研究する->計算機科学->の->一分野を->指す->語->研究分野とも->される
計算機科学->の->一分野を->指す->語->研究分野とも->される
される
される
一分野を->指す->語->研究分野とも->される
される
語->研究分野とも->される
言語の->推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
理解や->推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
推論->問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
問題解決などの->知的行動を->代わって->行わせる->技術または->研究分野とも->される
される
知的行動を->代わって->行わせる->技術または->研究分野とも->される
される
人間に->代わって->行わせる->技術または->研究分野とも->される
コンピューターに->行わせる->技術または->研究分野とも->される
技術または->研究分野とも->される
計算機->コンピュータによる->情報処理システムの->実現に関する->研究分野とも->される
される
コンピュータによる->情報処理システムの->実現に関する->研究分野とも->される
知的な->情報処理システムの->実現に関する->研究分野とも->される
情報処理システムの->実現に関する->研究分野とも->される
される
設計や->実現に関する->研究分野とも->される
実現に関する->研究分野とも->される
研究分野とも->される
"""