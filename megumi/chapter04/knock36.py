#36.頻度上位10語
#出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import knock30
from collections import Counter
import matplotlib.pyplot as plt

# 形態素解析の結果を取得
result = knock30.parse_neko()

# 単語の出現頻度をカウントするためのリスト
words = []

for line in result:
    for dic in line:
        words.append(dic["surface"])

# 単語の出現頻度をカウント
word_counter = Counter(words)

# 出現頻度の高い順に並べる
sorted_word_freq = word_counter.most_common(10)

# 出現頻度の高い10語とその頻度を取得
words, freqs = zip(*sorted_word_freq)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.bar(words, freqs, color='skyblue')
plt.xlabel('単語')
plt.ylabel('出現頻度')
plt.title('出現頻度が高い単語トップ10')
plt.xticks(rotation=45)
plt.show()
