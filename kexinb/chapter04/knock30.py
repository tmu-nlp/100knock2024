# task 30. 形態素解析結果の読み込み
# 形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
# ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
# 1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

# Dictionary used: mecab-ipadic-NEologd : Neologism dictionary for MeCab 
# https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.ja.md

# 表層形 \t 品詞,  品詞細分類1, 品詞細分類2, 品詞細分類3,活用型,  活用形, 原形,   読み,   発音
# 生れ	   動詞,   自立,         *,          *,     一段,    連用形, 生れる, ウマレ, ウマレ
# た	   助動詞,  *,          *,          *,     特殊・タ, 基本形, た,    タ,     タ

from typing import List, Dict

def parse_mecab(mecabData:str) -> List[Dict]: 
    morphemes = mecabData.strip().split('\n')
    result = []
    sentence = []

    for morph in morphemes:
        # Skip empty lines or reset and store the sentence at EOS
        if morph == "" or morph == "EOS":
            if sentence:
                result.append(sentence)
                sentence = []
            continue
        
        # Split the morpheme into surface and attributes
        surface, attrs = morph.split('\t')
        attrs = attrs.split(',')
                
        # Create a dictionary for the morpheme
        # assert len(attrs) >= 7
        morphDict = {
            'surface': surface,
            'base': attrs[6],
            'pos': attrs[0],
            'pos1': attrs[1]
        }
        sentence.append(morphDict)

    return result
        
if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        for line in output: 
            print(line)
        


# -------------------------------------------------------------------------------------------------------------------------
# def parse_sentence(sentence:str) -> List[Dict]:
#     result = []
#     morphemes = sentence.split('\n')
#     for morph in morphemes:
#         if len(morph) == 0: # skip empty lines
#             continue
#         else: # MOS
#             (surface, attrs) = morph.split('\t')
#             if surface == "":
#                 continue
#             attrs = attrs.split(',')
#             assert len(attrs) >= 7 # 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
#             morphDict = {
#                 'surface': surface,
#                 'base': attrs[6],
#                 'pos': attrs[0],
#                 'pos1': attrs[1]
#             }
#             result.append(morphDict)
#     return result

# if __name__ == "__main__":
#     with open('neko.txt.mecab','r') as f:
#         sentences = f.read().split('EOS\n')
#         for sentence in sentences:
#                 print(parse_sentence(sentence))