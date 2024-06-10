import zipfile
import cabocha
from graphviz import Digraph

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def is_symbol(self):
        return self.pos == '記号'

    def is_noun(self):
        return self.pos == '名詞'

    def is_verb(self):
        return self.pos == '動詞'

    def is_sahen_noun(self):
        return self.pos == '名詞' and self.pos1 == 'サ変接続'

    def is_particle(self):
        return self.pos == '助詞' and self.surface == 'を'

    def __repr__(self):
        return f"Morph(surface='{self.surface}', base='{self.base}', pos='{self.pos}', pos1='{self.pos1}')"

class Chunk:
    def __init__(self, morphs=None, dst=-1):
        self.morphs = morphs if morphs is not None else []
        self.dst = dst
        self.srcs = []

    def get_text(self, exclude_symbols=True):
        if exclude_symbols:
            return ''.join([morph.surface for morph in self.morphs if not morph.is_symbol()])
        else:
            return ''.join([morph.surface for morph in self.morphs])

    def has_noun(self):
        return any(morph.is_noun() for morph in self.morphs)

    def has_verb(self):
        return any(morph.is_verb() for morph in self.morphs)

    def get_base_verb(self):
        for morph in self.morphs:
            if morph.is_verb():
                return morph.base
        return None

    def get_case_particles(self):
        particles = []
        for morph in self.morphs:
            if morph.is_particle():
                particles.append(morph.surface)
        return particles

    def is_sahen_wo(self):
        for i in range(len(self.morphs) - 1):
            if self.morphs[i].is_sahen_noun() and self.morphs[i + 1].is_particle():
                return True
        return False

    def get_sahen_wo_text(self):
        for i in range(len(self.morphs) - 1):
            if self.morphs[i].is_sahen_noun() and self.morphs[i + 1].is_particle():
                return ''.join([morph.surface for morph in self.morphs[:i + 2]])
        return None

    def __repr__(self):
        morphs_str = ''.join([morph.surface for morph in self.morphs])
        return f"Chunk(morphs='{morphs_str}', dst={self.dst}, srcs={self.srcs})"

def parse_cabocha_output_with_chunks(file_path):
    sentences = []
    chunks = []
    chunk = None
    idx_to_chunk = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('*'):  # 係り受け解析情報の行
                if chunk is not None:
                    chunks.append(chunk)
                chunk_info = line.split(' ')
                chunk_idx = int(chunk_info[1])
                dst = int(chunk_info[2].rstrip('D'))
                chunk = Chunk(dst=dst)
                idx_to_chunk[chunk_idx] = chunk
            elif line.strip() == 'EOS':
                if chunk is not None:
                    chunks.append(chunk)
                if chunks:
                    for i, c in enumerate(chunks):
                        if c.dst != -1:
                            chunks[c.dst].srcs.append(i)
                    sentences.append(chunks)
                chunks = []
                chunk = None
                idx_to_chunk = {}
            else:
                fields = line.split('\t')
                if len(fields) == 2:
                    surface = fields[0]
                    other_info = fields[1].split(',')
                    base = other_info[6]
                    pos = other_info[0]
                    pos1 = other_info[1]
                    morph = Morph(surface, base, pos, pos1)
                    chunk.morphs.append(morph)
    
    return sentences

def extract_sahen_wo_patterns(sentences):
    patterns = []
    for chunks in sentences:
        for chunk in chunks:
            if chunk.is_sahen_wo() and chunk.dst != -1:
                base_verb = chunks[chunk.dst].get_base_verb()
                sahen_wo_text = chunk.get_sahen_wo_text()
                if base_verb and sahen_wo_text:
                    predicate = sahen_wo_text + base_verb
                    particles_and_phrases = []
                    for src in chunks[chunk.dst].srcs:
                        if src != chunks.index(chunk):  # 自分自身を除く
                            particles = chunks[src].get_case_particles()
                            if particles:
                                particles_and_phrases.append((particles[0], chunks[src].get_text()))
                    if particles_and_phrases:
                        particles_and_phrases = sorted(particles_and_phrases)  # 助詞でソート
                        particles = ' '.join([p[0] for p in particles_and_phrases])
                        phrases = ' '.join([p[1] for p in particles_and_phrases])
                        patterns.append((predicate, particles, phrases))
    return patterns

# 解析結果のファイルパス
parsed_file_path = 'ai.ja.txt.parsed'

# 解析結果を読み込む
sentences_with_chunks = parse_cabocha_output_with_chunks(parsed_file_path)

# サ変接続名詞+を+動詞の格パターンを抽出
patterns = extract_sahen_wo_patterns(sentences_with_chunks)

# 結果をファイルに保存
with open('sahen_wo_patterns.txt', 'w', encoding='utf-8') as file:
    for predicate, particles, phrases in patterns:
        file.write(f"{predicate}\t{particles}\t{phrases}\n")

# コーパス中で頻出する述語と格パターンの組み合わせ
# sort sahen_wo_patterns.txt | uniq -c | sort -nr | head

# 特定の動詞の格パターン
# grep '^行う' sahen_wo_patterns.txt | sort | uniq -c | sort -nr
# grep '^なる' sahen_wo_patterns.txt | sort | uniq -c | sort -nr
# grep '^与える' sahen_wo_patterns.txt | sort | uniq -c | sort -nr