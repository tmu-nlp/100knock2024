import matplotlib.pyplot as plt
import japanize_matplotlib
from knock35 import parse_neko, count_word_frequency, sort_word_frequency

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    word_freq = count_word_frequency(sentences)

    freq_list = list(word_freq.values())

    plt.figure(figsize=(10, 6))
    plt.hist(freq_list, bins=100)
    plt.xlabel('出現頻度')
    plt.ylabel('単語の種類数')
    plt.title('単語の出現頻度のヒストグラム')
    plt.tight_layout()

    plt.savefig('knock38.png')
    plt.show()