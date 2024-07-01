#34.名詞の連接
#名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．
#最長一致：ここでは、最も連続する名詞を指す。



def parse_neko():
    result = []
    sentence = []

    with open("neko.txt.mecab", encoding='utf-8') as f:
        for line in f:
            l1 = line.split("\t")
            if len(l1) == 2:
                l2 = l1[1].split(",")
                morph = {
                    "surface": l1[0],
                    "base": l2[6],
                    "pos": l2[0],
                    "pos1": l2[1]
                }
                sentence.append(morph)
                if l2[1] == "句点":
                    result.append(sentence)
                    sentence = []

    return result

#名詞の連接を最長一致で抽出する関数の定義
def extract_longest_noun_sequences(parsed_text):
    longest_sequences = []

    for sentence in parsed_text:
        current_sequence = []
        for morph in sentence:
            if morph["pos"] == "名詞":
                current_sequence.append(morph["surface"])
            else:
                if len(current_sequence) > 1:
                    longest_sequences.append("".join(current_sequence))
                current_sequence = []
        if len(current_sequence) > 1:
            longest_sequences.append("".join(current_sequence))

    return longest_sequences

# 形態素解析を実行
parsed_text = parse_neko()

# 名詞の連接を最長一致で抽出
longest_noun_sequences = extract_longest_noun_sequences(parsed_text)

# 抽出結果を表示
for sequence in longest_noun_sequences:
    print(sequence)

"""
出力結果
——おい苦沙弥先生
独仙君
万年漬
後ろ向
迷亭君
独仙君
東風君
寒月君
"""