import zipfile
import cabocha

class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __repr__(self):
        return f"Morph(surface='{self.surface}', base='{self.base}', pos='{self.pos}', pos1='{self.pos1}')"

class Chunk:
    def __init__(self, morphs=None, dst=-1):
        self.morphs = morphs if morphs is not None else []
        self.dst = dst
        self.srcs = []

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

# 冒頭の説明文の文節の文字列と係り先を表示
for chunk in sentences_with_chunks[0]:  # 最初の文
    morphs_str = ''.join([morph.surface for morph in chunk.morphs])
    print(f"{morphs_str}\t{chunk.dst}")
