# task49. 名詞間の係り受けパスの抽出
"""
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ
ただし名詞句ペアの文節番号がiとj(i<j)のとき，係り受けパスは以下の仕様を満たすものとする.

パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
文節iとjに含まれる名詞句はそれぞれ,XとYに置換する
また,係り受けパスの形状は,以下の2通りが考えられる.
文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
上記以外で,文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 
    文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス,文節kの内容を” | “で連結して表示
"""

############## BUGGY #################
from itertools import combinations
from knock40 import load_file
from knock47 import Chunk, parse_chunk

def chunk_to_text(chunk, replace_noun=None):
    text = ''
    replaced = False
    for morph in chunk.morphs:
        if replace_noun and morph.pos == "名詞" and not replaced:
            text += replace_noun
            replaced = True
        else:
            text += morph.surface
    return text

def extract_paths(sentence):
    results = []
    nouns = []
    for i, chunk in enumerate(sentence):
        for morph in chunk.morphs:
            if morph.pos == "名詞":
                nouns.append(i)
                break

    for i, j in combinations(nouns, 2):
        path_i = []
        path_j = []

        orig_i, orig_j = i, j

        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence[i].dst
            else:
                path_j.append(j)
                j = sentence[j].dst

        if len(path_j) == 0:
            chunkX = chunk_to_text(sentence[orig_i], replace_noun="X")
            chunkY = chunk_to_text(sentence[i], replace_noun="Y")
            path_i_text = [chunkX] + [chunk_to_text(sentence[k]) for k in path_i[1:]] + [chunkY]
            results.append(' -> '.join(path_i_text))
        else:
            chunkX = chunk_to_text(sentence[orig_i], replace_noun="X")
            chunkY = chunk_to_text(sentence[orig_j], replace_noun="Y")
            chunk_k = chunk_to_text(sentence[i])

            path_i_text = [chunkX] + [chunk_to_text(sentence[k]) for k in path_i[1:]]
            path_j_text = [chunkY] + [chunk_to_text(sentence[k]) for k in reversed(path_j)]

            results.append(' -> '.join(path_i_text) + ' | ' + ' -> '.join(path_j_text) + ' | ' + chunk_k)
    
    return results

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]

        for sentence in chunked_sentences:
            results = extract_paths(sentence)
            for result in results:
                print(result)