# task48. 名詞から根へのパスの抽出
"""
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ.
ただし，構文木上のパスは以下の仕様を満たすものとする．
- 各文節は（表層形の）形態素列で表現する
- パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する
e.g.「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
        ジョンマッカーシーは -> 作り出した
        AIに関する -> 最初の -> 会議で -> 作り出した
        最初の -> 会議で -> 作り出した
        会議で -> 作り出した
        人工知能という -> 用語を -> 作り出した
        用語を -> 作り出した
"""

from knock40 import load_file
from knock47 import Chunk, parse_chunk

def extract_paths_to_root(sentence): # 
    results = []
    for chunk in sentence:
        if any(morph.pos == "名詞" for morph in chunk.morphs):
            path = [chunk.chunk_to_text()]
            dst = chunk.dst
            while dst != -1:
                chunk = sentence[dst]
                path.append(chunk.chunk_to_text())
                dst = chunk.dst
            results.append(" -> ".join(path))
    return results


if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]

        for sentence in chunked_sentences:
            paths = extract_paths_to_root(sentence)
            for path in paths:
                print(path)
