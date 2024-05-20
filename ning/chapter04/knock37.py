import matplotlib.pyplot as plt
from collections import Counter
import re
import japanize_matplotlib

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

#記号、助詞、助動詞を除外し、猫と共起する単語をカウント
def count_cooccurrences(sentences, target_word):
    cooccurrence_counter = Counter()
    for sentence in sentences:
        words_in_sentence = [morph['surface'] for morph in sentence]
        if target_word in words_in_sentence:
            for morph in sentence:
                if morph['surface'] != target_word and morph['pos'] not in ['記号', '助詞', '助動詞']:
                    cooccurrence_counter[morph['surface']] += 1
    return cooccurrence_counter

file_path = 'neko.txt.mecab'
sentences = read_mecab(file_path)
cooccurrence_counter = count_cooccurrences(sentences, '猫')

#出現頻度上位10語の抽出
most_common_words = cooccurrence_counter.most_common(10)

words, counts = zip(*most_common_words)
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xlabel('単語')
plt.ylabel('出現頻度')
plt.title('「猫」と共起する単語の出現頻度上位10')
plt.xticks(rotation=45)
plt.show()
