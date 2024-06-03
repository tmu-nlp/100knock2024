import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import japanize_matplotlib

#形態素解析結果の読み込み
def read_mecab(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
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
    return sentences

#全単語の出現頻度をカウント
def count_word_frequencies(sentences):
    word_counter = Counter()
    for sentence in sentences:
        for morph in sentence:
            if morph['pos'] not in ['記号', '助詞', '助動詞']:
                word_counter[morph['surface']] += 1
    return word_counter

file_path = 'neko.txt.mecab'
sentences = read_mecab(file_path)
word_counter = count_word_frequencies(sentences)

#出現頻度順位とその出現頻度を取得
frequencies = [freq for word, freq in word_counter.most_common()]
ranks = range(1, len(frequencies) + 1)

#両対数グラフのプロット
plt.figure(figsize=(10, 5))
plt.loglog(ranks, frequencies, marker='o')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.title('単語の出現頻度順位と出現頻度の両対数グラフ')
plt.grid(True, which="both", ls="--")
plt.show()
