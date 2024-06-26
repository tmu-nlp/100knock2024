import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import defaultdict
from knock30 import parse_neko

def count_co_occurrence(sentences, target_word):
    co_occur_freq = defaultdict(int)
    for sentence in sentences:
        words = [morph['base'] for morph in sentence if morph['pos'] != '記号']
        if target_word in words:
            for word in words:
                if word != target_word:
                    co_occur_freq[word] += 1
    return co_occur_freq

def sort_co_occurrence(co_occur_freq):
    return sorted(co_occur_freq.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab')
    target_word = '猫'
    co_occur_freq = count_co_occurrence(sentences, target_word)
    sorted_co_occur = sort_co_occurrence(co_occur_freq)

    top10_words = [word for word, freq in sorted_co_occur[:10]]
    top10_freqs = [freq for word, freq in sorted_co_occur[:10]]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top10_words)), top10_freqs)
    plt.xticks(range(len(top10_words)), top10_words, rotation=45, ha='right')
    plt.xlabel('単語')
    plt.ylabel('共起頻度')
    plt.title(f'「{target_word}」との共起頻度上位10語')
    plt.tight_layout()

    plt.savefig('knock37.png')
    plt.show()