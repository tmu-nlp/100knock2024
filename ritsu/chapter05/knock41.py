from knock40 import Morph, parse_morphs

class Chunk:
    """ 文節を表すクラス """
    def __init__(self):
        self.morphs = []  # Morphオブジェクトのリスト
        self.dst = -1     # 係り先文節インデックス番号
        self.srcs = []    # 係り元文節インデックス番号のリスト

    def __str__(self):
        """ 文節の情報を文字列として返す """
        return ''.join([morph.surface for morph in self.morphs if morph.pos != '記号'])

def parse_chunks(file_path):
    """ 係り受け解析結果ファイルからChunkオブジェクトのリストのリストを生成する関数 """
    sentences = []
    chunks = []
    current_chunk = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('*'):
                if current_chunk is not None:
                    chunks.append(current_chunk)
                current_chunk = Chunk()
                current_chunk.dst = int(line.split(' ')[2].strip('D'))
            elif line == 'EOS\n':
                if current_chunk is not None:
                    chunks.append(current_chunk)
                if chunks:
                    # 係り元の更新
                    for i, chunk in enumerate(chunks):
                        if chunk.dst != -1:
                            chunks[chunk.dst].srcs.append(i)
                    sentences.append(chunks)
                current_chunk = None
                chunks = []
            else:
                if current_chunk is not None:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        surface = parts[0]
                        details = parts[1].split(',')
                        morph = Morph(surface, details[6], details[0], details[1])
                        current_chunk.morphs.append(morph)
    return sentences

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = list(parse_chunks(file_path))
    with open('knock41.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for chunk in sentence:
                f.write(f"{str(chunk)} -> dst: {chunk.dst}, srcs: {chunk.srcs}\n")

if __name__ == "__main__":
    main()
