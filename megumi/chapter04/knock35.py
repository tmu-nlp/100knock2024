#35.単語の出現頻度
#文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

import knock30
from collections import Counter

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
sorted_word_freq = word_counter.most_common()

# 出現頻度の高い順に単語とその頻度を表示
print("出現頻度の高い単語トップ5:")
for word, freq in sorted_word_freq[:5]:
    print(f"{word}: {freq}")
