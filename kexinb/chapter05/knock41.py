# task41. 係り受け解析結果の読み込み（文節・係り受け）
# 40に加えて，文節を表すクラスChunkを実装せよ

# このクラスは
    # 形態素（Morphオブジェクト）のリスト（morphs）
    # 係り先文節インデックス番号（dst）
    # 係り元文節インデックス番号のリスト（srcs）をメンバ変数に持つこととする
# さらに，入力テキストの係り受け解析結果を読み込み
# １文をChunkオブジェクトのリストとして表現し，冒頭の説明文の文節の文字列と係り先を表示せよ
# 本章の残りの問題では，ここで作ったプログラムを活用せよ．

# *    0            17D                            1/1                          0.388993
# * 文節番号  係り先の文節番号(係り先なし:-1D)   主辞の形態素番号/機能語の形態素番号   係り関係のスコア(大きい方が係りやすい)

from collections import defaultdict
from knock40 import Morph, load_file

class Chunk:
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs # List(Morph)
        self.dst = dst
        self.srcs = srcs

def parse_chunk(sentence): # return List(Chunk)
    result = [] # collect all chunks in the sentence
    
    morphs = [] # collect all morphs in a chunk
    curr_ind = -1  # init to an invalid index
    srcs_dict = defaultdict(list) # {dst:[srcs]}
    lines = sentence.split("\n")
    
    for line in lines:
        if len(line) == 0: # EOS
            continue
        elif line[0] == "*":  # if header: enter new chunk
            if curr_ind != -1: # if not first chunk
                result.append(Chunk(morphs, dst, srcs_dict[curr_ind])) # append previous chunk to result
                morphs = [] # reset morphs
            chunk_info =  line.split()
            curr_ind = int(chunk_info[1])
            dst = int(chunk_info[2].rstrip('D'))
            srcs_dict[dst].append(curr_ind)
        else:
            morphs.append(Morph(line))
    if len(morphs) != 0:
        result.append(Chunk(morphs, dst, srcs_dict[curr_ind]))
    
    return result

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]
        
        for chunk in chunked_sentences[1]:
            print([m.surface for m in chunk.morphs], chunk.dst, chunk.srcs)
