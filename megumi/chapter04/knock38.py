#38.ヒストグラムPermalink
#単語の出現頻度のヒストグラムを描け．
# ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．
# 縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

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

# 出現頻度のリストを作成
frequencies = list(word_counter.values())

# 出現頻度の最大値を取得
max_frequency = max(frequencies)

# 出現頻度ごとの単語の種類数をカウント
frequency_counts = Counter(frequencies)

# ヒストグラムの描画
plt.figure(figsize=(10, 6))
plt.bar(frequency_counts.keys(), frequency_counts.values(), edgecolor='black')
plt.xlabel('出現頻度')
plt.ylabel('単語の種類数')
plt.title('単語の出現頻度のヒストグラム')
plt.xticks(range(1, max_frequency + 1))
plt.show()



