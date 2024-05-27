# task42. 係り元と係り先の文節の表示
# 係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

from knock40 import load_file
from knock41 import Chunk, parse_chunk

def chunk_to_text(self):
    return ''.join(m.surface for m in self.morphs if m.pos != "記号")

Chunk.chunk_to_text = chunk_to_text

def print_dst(sentence): # List(Chunk) -> List[str]
    result = []
    for chunk in sentence:
        if chunk.dst == -1:
            result.append(chunk.chunk_to_text() + "\tRoot")
        else:
            dst_chunk = sentence[chunk.dst]
            result.append(chunk.chunk_to_text() + "\t" + \
                        dst_chunk.chunk_to_text())
    return result

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]
        for sentence in chunked_sentences:
            output = print_dst(sentence)
            print(*output, sep="\n")