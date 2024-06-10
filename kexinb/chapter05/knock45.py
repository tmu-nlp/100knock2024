# task45.動詞の格パターンの抽出
"""
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． 
ただし，出力は以下の仕様を満たすようにせよ．
- 動詞を含む文節において，最左の動詞の基本形を述語とする
- 述語に係る助詞を格とする
- 述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

eg. 「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
    「作り出す」-「ジョン・マッカーシーは」，「会議で」，「用語を」
    -> 作り出す    で は を
"""

from knock40 import load_file
from knock42 import Chunk, parse_chunk

# Extend Chunk class
def get_base_of_first_verb(self): # get base form of the first verb in a chunk
        for morph in self.morphs:
            if morph.pos == "動詞":
                return morph.base
        return None
    
def get_surface_of_pos(self, pos): # get all surface of a certain pos in a chunk as a list
    particles = []
    for morph in self.morphs:
        if morph.pos == pos:
            particles.append(morph.surface)
    return particles

Chunk.get_base_of_first_verb = get_base_of_first_verb
Chunk.get_surface_of_pos = get_surface_of_pos

def extract_verb_pos(sentences, pos): # get all (pos) with dst containing verb
    result = []
    for sentence in sentences:
        for chunk in sentence:
            verb_base = chunk.get_base_of_first_verb()
            if verb_base:
                particles = []
                for src in chunk.srcs:
                    particles.extend(sentence[src].get_surface_of_pos(pos))
                if particles:
                    particles = sorted(set(particles))
                    result.append(f"{verb_base}\t{' '.join(particles)}")
    return result


if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]

        results = extract_verb_pos(chunked_sentences, "助詞")
        for result in results:
            print(result)