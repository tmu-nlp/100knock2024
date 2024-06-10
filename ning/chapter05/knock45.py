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

    def is_particle(self):
        return self.pos == '助詞' and self.pos1 == '格助詞'

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

def extract_predicate_case_patterns(sentences):
    patterns = []
    for chunks in sentences:
        for chunk in chunks:
            if chunk.has_verb():
                base_verb = chunk.get_base_verb()
                particles = []
                for src in chunk.srcs:
                    particles += chunks[src].get_case_particles()
                if base_verb and particles:
                    particles = sorted(set(particles))  # 重複除去してソート
                    patterns.append((base_verb, ' '.join(particles)))
    return patterns

# 解析結果のファイルパス
parsed_file_path = 'ai.ja.txt.parsed'

# 解析結果を読み込む
sentences_with_chunks = parse_cabocha_output_with_chunks(parsed_file_path)

# 述語と格パターンを抽出
patterns = extract_predicate_case_patterns(sentences_with_chunks)

# 結果をファイルに保存
with open('predicate_case_patterns.txt', 'w', encoding='utf-8') as file:
    for verb, particles in patterns:
        file.write(f"{verb}\t{particles}\n")

# コーパス中で頻出する述語と格パターンの組み合わせ
# sort predicate_case_patterns.txt | uniq -c | sort -nr | head

# 特定の動詞の格パターン
# grep '^行う' predicate_case_patterns.txt | sort | uniq -c | sort -nr
# grep '^なる' predicate_case_patterns.txt | sort | uniq -c | sort -nr
# grep '^与える' predicate_case_patterns.txt | sort | uniq -c | sort -nr