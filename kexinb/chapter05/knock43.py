# task43. 名詞を含む文節が動詞を含む文節に係るものを抽出
# 名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ.
# ただし，句読点などの記号は出力しないようにせよ．

from knock40 import load_file
from knock41 import parse_chunk
from knock42 import Chunk

def print_dst_constr(sentence, pos1, pos2): # List(Chunk) -> List[str]
    result = []
    print_flag = False

    for chunk in sentence:
        dst_chunk = sentence[chunk.dst]
        print_flag = (pos1 in [m.pos for m in chunk.morphs]) & \
            (pos2 in [m.pos for m in dst_chunk.morphs])
        if print_flag:
            result.append(chunk.chunk_to_text() + "\t" + \
                        dst_chunk.chunk_to_text())
    return result

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]
        for sentence in chunked_sentences:
            output = print_dst_constr(sentence, "名詞", "動詞")
            if output: # skip empty lines
                print(*output, sep="\n")