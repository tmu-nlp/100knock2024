class Morph:
    """ 形態素を表すクラス """
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface  # 表層形
        self.base = base        # 基本形
        self.pos = pos          # 品詞
        self.pos1 = pos1        # 品詞細分類1

    def __str__(self):
        return f"surface: {self.surface}, base: {self.base}, pos: {self.pos}, pos1: {self.pos1}"

def parse_morphs(file_path):
    """ 係り受け解析結果ファイルからMorphオブジェクトのリストを生成する関数 """
    morphs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line != 'EOS\n' and not line.startswith('*'):
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    surface = parts[0]
                    details = parts[1].split(',')
                    morph = Morph(surface, details[6], details[0], details[1])
                    morphs.append(morph)
    return morphs

def main():
    # ai.ja.txt.parsedのパスは適宜調整してください。
    file_path = 'ai.ja.txt.parsed'
    morphs = parse_morphs(file_path)
    with open('knock40.txt', 'w', encoding='utf-8') as f:
        for morph in morphs:
            f.write(str(morph) + '\n')

if __name__ == "__main__":
    main()
