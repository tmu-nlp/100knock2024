class Morph(): 
    def __init__(self, morph):
        surface, attr = morph.split('\t')
        attr_list = attr.split(',')
        self.surface = surface  # 表層系
        self.base = attr_list[6]  # 原型
        self.pos = attr_list[0]  # 品詞
        self.pos1 = attr_list[1]  # 品詞細分類1

#sentencesの中に1つの文のMorphオブジェクトが入ったリストを入れる

sentences = [] #全てのsentenceリストを入れるリスト
morphs = [] #1つの文の全てのMorphクラスを入れるリスト

with open("ai.ja.txt.parsed", "r") as f:
    for line in f:
        if line[0] == '*':  # 今回はチャンクではなく文単位のリストなので、開幕が*の場合は飛ばす
            continue
        elif line == "EOS\n":  # EOS：文末
            if len(morphs) > 0:  # morphsが1以上ある場合
                sentences.append(morphs)  # sentencesにmorphsのリストを追加
            morphs = []  # 初期化
        else:
            morphs.append(Morph(line))  # 文が終わるまで、morphsにMorphクラスを追加

if __name__ == '__main__':
    for mophes in sentences[1]:
        print(vars(mophes))  # 辞書型を一気に出力