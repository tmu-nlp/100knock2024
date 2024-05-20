import matplotlib.pyplot as plt
import japanize_matplotlib
from knock35 import parse_neko, count_word_frequency, sort_word_frequency

if __name__ == "__main__":
    sentences = parse_neko('neko.txt.mecab') # neko.txt.mecabを形態素解析してリストを取得
    word_freq = count_word_frequency(sentences) # 単語の出現頻度を計算
    sorted_word_freq = sort_word_frequency(word_freq) # 出現頻度の高い順にソート

    ranks = range(1, len(sorted_word_freq) + 1) # 1から始まる連続した整数の範囲
    freqs = [freq for _, freq in sorted_word_freq] # 出現頻度のみを取り出したリスト 

    plt.figure(figsize=(10, 6))
    plt.scatter(ranks, freqs, s=10)
    plt.xscale('log') # x軸を対数スケールに
    plt.yscale('log') # y軸を対数スケールに
    plt.xlabel('出現頻度順位') 
    plt.ylabel('出現頻度')
    plt.title('Zipfの法則')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('knock39.png')
    plt.show()