# task 40. 係り受け解析結果の読み込み（形態素）
# 形態素を表すクラスMorphを実装せよ．このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．\
# さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，\
# 各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．

class Morph:
     def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')

        self.surface = surface
        self.base = attr_list[6]
        self.pos = attr_list[0]
        self.pos1 = attr_list[1]

def load_file(f): # return list of sentences
    text = f.read().split("EOS\n") # split by EOS\n
    text = [x for x in text if x] # get rid of empty lines
    return text

def parse_cabocha(sentence): # return List(Morph)
    result = []
    lines = sentence.split("\n")
    for line in lines:
        if len(line) == 0: # EOS
            continue
        elif line[0] == "*": # new sentence
            continue
        result.append(Morph(line))
    return result

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        # print(*text, sep="\n")
        morphemes = [parse_cabocha(sentence) for sentence in text]
        
        for morpheme in morphemes[1]:
           print(vars(morpheme))