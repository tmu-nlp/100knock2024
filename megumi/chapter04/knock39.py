#39. Zipfの法則
#単語の出現頻度順位を横軸，
# その出現頻度を縦軸として，両対数グラフをプロットせよ

import matplotlib.pyplot as plt
from collections import Counter
import knock30  # knock30.pyファイルをインポート

# 形態素解析の結果を取得
result = knock30.parse_neko()

# 単語の出現頻度をカウントするためのリスト
words = []

for line in result:
    for dic in line:
        words.append(dic["surface"])

# 単語の出現頻度をカウント
word_counter = Counter(words)

# 出現頻度のリストを作成し、頻度の高い順にソート
frequencies = list(word_counter.values())
frequencies.sort(reverse=True)

# 頻度順位（1位から順に）を作成
ranks = range(1, len(frequencies) + 1)

# 両対数グラフの描画
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker="o", linestyle="none")
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.title('単語の出現頻度順位と出現頻度の両対数グラフ')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()
