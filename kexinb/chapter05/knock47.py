# task47. 機能動詞構文のマイニング
# 動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい.
# 学習を行う	に を	元に 経験を

from knock40 import load_file
from knock46 import Chunk, parse_chunk

def get_sahen_wo(self):
    for i in range(len(self.morphs) - 1):
        if len(self.morphs) == 2 and\
            self.morphs[i].pos1 == "サ変接続" and \
            self.morphs[i + 1].surface == "を":
            return ''.join([self.morphs[i].surface, self.morphs[i + 1].surface])
    return None

Chunk.get_sahen_wo = get_sahen_wo

def extract_sahen_pos_phrase(sentences, pos):
    results = []
    for sentence in sentences:
        for chunk in sentence:
            sahen_wo = chunk.get_sahen_wo()
            if sahen_wo and chunk.dst != -1:
                dst_chunk = sentence[chunk.dst]
                verb_base = dst_chunk.get_base_of_first_verb()
                if verb_base:
                    predicate = sahen_wo + verb_base
                    particles = []
                    phrases = []
                    for src in dst_chunk.srcs:
                        if src != sentence.index(chunk):
                            src_particles, src_phrases = sentence[src].get_surface_of_pos_phrases(pos)
                            particles.extend(src_particles)
                            phrases.extend(src_phrases)
                    if particles:
                        sorted_particles_phrases = sorted(zip(particles, phrases))
                        particles, phrases = zip(*sorted_particles_phrases)
                        result = f"{predicate}\t{' '.join(particles)}\t{' '.join(phrases)}"
                        results.append(result)
    return results

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]

        results = extract_sahen_pos_phrase(chunked_sentences, "助詞")
        for result in results:
            print(result)