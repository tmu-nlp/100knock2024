# task46. 動詞の格フレーム情報の抽出
"""
eg. 「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
    「作り出す」-「ジョン・マッカーシーは」，「会議で」，「用語を」
    -> 作り出す    で は を	会議で ジョンマッカーシーは 用語を
"""

from knock40 import load_file
from knock45 import Chunk, parse_chunk

def get_surface_of_pos_phrases(self, pos):
    particles = []
    phrases = []
    for morph in self.morphs:
        if morph.pos == pos:
            particles.append(morph.surface)
            phrases.append(self.chunk_to_text())
    return particles, phrases

Chunk.get_surface_of_pos_phrases = get_surface_of_pos_phrases


def extract_verb_pos_phrase(sentences, pos):
    results = []
    for sentence in sentences:
        for chunk in sentence:
            verb_base = chunk.get_base_of_first_verb()
            if verb_base:
                particles = []
                phrases = []
                for src in chunk.srcs:
                    src_particles, src_phrases = \
                        sentence[src].get_surface_of_pos_phrases(pos)
                    particles.extend(src_particles)
                    phrases.extend(src_phrases)
                if particles:
                    sorted_particles_phrases = sorted(zip(particles, phrases))
                    particles, phrases = zip(*sorted_particles_phrases)
                    result = f"{verb_base}\t{' '.join(particles)}\t{' '.join(phrases)}"
                    results.append(result)
    return results

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]

        results = extract_verb_pos_phrase(chunked_sentences, "助詞")
        for result in results:
            print(result)