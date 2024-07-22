#37.猫」と共起頻度の高い上位10語
#「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import MeCab
from collections import Counter
import matplotlib.pyplot as plt

# 形態素解析器の初期化
mecab = MeCab.Tagger()

# テキストデータの読み込み
with open('neko.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 形態素解析を行い、単語に分割
node = mecab.parseToNode(text)
words = []
while node:
    if node.feature.split(',')[0] != 'BOS/EOS':
        words.append(node.surface)
    node = node.next

# 「猫」と共起する単語をカウント
co_occurrence_counter = Counter()
for i, word in enumerate(words):
    if word == "猫":
        # 「猫」の前後の単語をカウント
        if i > 0:
            co_occurrence_counter[words[i-1]] += 1
        if i < len(words) - 1:
            co_occurrence_counter[words[i+1]] += 1

# 共起頻度の高い順に並べる
sorted_co_occurrence = co_occurrence_counter.most_common(10)

# 共起頻度の高い10語とその頻度を取得
co_words, co_freqs = zip(*sorted_co_occurrence)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.bar(co_words, co_freqs, color='skyblue')
plt.xlabel('単語')
plt.ylabel('共起頻度')
plt.title('「猫」と共起頻度が高い単語トップ10')
plt.xticks(rotation=45)
plt.show()
