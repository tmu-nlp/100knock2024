#全部を載せると差異が見れないため、出現頻度が30未満の単語だけにした

import matplotlib.pyplot as plt
from collections import Counter
import japanize_matplotlib

file_path = 'neko.txt.mecab'

#形態素解析結果neko.txt.mecabの読み込み
sentences = []
with open(file_path, 'r', encoding='utf-8') as f:
    sentence = []
    for line in f:
        if line == 'EOS\n':
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            surface, feature = line.split('\t')
            features = feature.split(',')
            morph = {
                'surface': surface,
                'base': features[6],
                'pos': features[0],
                'pos1': features[1]
            }
            sentence.append(morph)

#各単語の出現頻度をカウント
word_counter = Counter()
for sentence in sentences:#品詞が「記号」、「助詞」、「助動詞」でない単語のみをカウント
    for morph in sentence:
        if morph['pos'] not in ['記号', '助詞', '助動詞']:
            word_counter[morph['surface']] += 1

#出現頻度が30未満の単語のみをリストに抽出
frequencies = [freq for freq in word_counter.values() if freq < 30]

#ヒストグラムの作成
plt.figure(figsize=(10, 5))
plt.hist(frequencies, bins=range(1, 30), edgecolor='black', align='left')
plt.xlabel('出現頻度')
plt.ylabel('単語の種類数')
plt.title('出現頻度ヒストグラム')
plt.show()

