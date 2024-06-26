from knock30 import parse_neko
from collections import defaultdict

def count_word_frequency(sentences):
    word_freq = defaultdict(int) # 単語の出現頻度を格納するdefaultdict(int)型
    for sentence in sentences:
        for morph in sentence:
            if morph['pos'] != '記号':
                word_freq[morph['base']] += 1 # 単語の出現頻度をカウント
    return word_freq # 単語の出現頻度を返す, defaultdict(int)型

def sort_word_frequency(word_freq):
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True) # 単語の出現頻度を降順にソートして返す

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    word_freq = count_word_frequency(sentences)
    sorted_word_freq = sort_word_frequency(word_freq)
    print(sorted_word_freq[:10])  # 出現頻度の高い上位10語を表示