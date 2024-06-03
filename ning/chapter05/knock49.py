import zipfile
import cabocha

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

    def replace_noun_with(self, replacement):
        new_text = []
        replaced = False
        for morph in self.morphs:
            if morph.is_noun() and not replaced:
                new_text.append(replacement)
                replaced = True
            else:
                new_text.append(morph.surface)
        return ''.join(new_text)

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

def extract_noun_paths(sentences):
    paths = []
    for chunks in sentences:
        for i, chunk_i in enumerate(chunks):
            if chunk_i.has_noun():
                for j in range(i + 1, len(chunks)):
                    chunk_j = chunks[j]
                    if chunk_j.has_noun():
                        path = []
                        current_chunk = chunk_i
                        while current_chunk is not None:
                            path.append(current_chunk)
                            if current_chunk.dst == j:
                                break
                            elif current_chunk.dst != -1:
                                current_chunk = chunks[current_chunk.dst]
                            else:
                                current_chunk = None
                        if current_chunk is not None:
                            path_text = ' -> '.join([c.replace_noun_with('X') if c == chunk_i else (c.replace_noun_with('Y') if c == chunk_j else c.get_text()) for c in path])
                            paths.append(path_text)
                        else:
                            path_i = []
                            path_j = []
                            common_chunk = None
                            current_chunk = chunk_i
                            while current_chunk is not None:
                                path_i.append(current_chunk)
                                if current_chunk.dst != -1:
                                    current_chunk = chunks[current_chunk.dst]
                                else:
                                    current_chunk = None
                            current_chunk = chunk_j
                            while current_chunk is not None:
                                path_j.append(current_chunk)
                                if current_chunk.dst != -1:
                                    current_chunk = chunks[current_chunk.dst]
                                else:
                                    current_chunk = None
                            for c_i in path_i:
                                if c_i in path_j:
                                    common_chunk = c_i
                                    break
                            if common_chunk is not None:
                                path_i = path_i[:path_i.index(common_chunk)]
                                path_j = path_j[:path_j.index(common_chunk)]
                                path_text = ' -> '.join([c.replace_noun_with('X') for c in path_i]) + ' | ' + ' -> '.join([c.get_text() for c in path_j]) + ' | ' + common_chunk.get_text()
                                paths.append(path_text)
    return paths

# ZIPファイルのパス
zip_file_path = 'ai.ja.zip'

# 解凍先ディレクトリ
extract_dir = './extracted_files'

# ZIPファイルの解凍
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 解凍されたファイルのパス
text_file_path = f'{extract_dir}/ai.ja.txt'

# テキストファイルの読み込み
with open(text_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# CaboChaの初期化
c = cabocha.Parser()

# 解析結果を保存するファイルのパス
parsed_file_path = 'ai.ja.txt.parsed'

# テキストを行ごとに解析
with open(parsed_file_path, 'w', encoding='utf-8') as parsed_file:
    for line in text.split('\n'):
        if line.strip():  # 空行は無視
            tree = c.parse(line)
            parsed_file.write(tree.toString(cabocha.FORMAT_LATTICE))
            parsed_file.write('\n')

# 解析結果を読み込む
sentences_with_chunks = parse_cabocha_output_with_chunks(parsed_file_path)

# 名詞間の係り受けパスを抽出
paths = extract_noun_paths(sentences_with_chunks)

# 結果を表示
for path in paths:
    print(path)

print("解析が完了しました。")
