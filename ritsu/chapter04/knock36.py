import matplotlib.pyplot as plt
import japanize_matplotlib
from knock35 import parse_neko, count_word_frequency, sort_word_frequency

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    word_freq = count_word_frequency(sentences)
    sorted_word_freq = sort_word_frequency(word_freq)

    top10_words = [word for word, freq in sorted_word_freq[:10]]
    top10_freqs = [freq for word, freq in sorted_word_freq[:10]]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top10_words)), top10_freqs)
    plt.xticks(range(len(top10_words)), top10_words, rotation=45, ha='right')
    plt.xlabel('単語')
    plt.ylabel('出現頻度')
    plt.title('出現頻度上位10語')
    plt.tight_layout()
    plt.savefig('knock36.png')
    plt.show()