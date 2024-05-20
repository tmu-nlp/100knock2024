#句読点などの記号は除外。助詞と助動詞なども除外。

from collections import Counter
import matplotlib.pyplot as plt
import japanize_matplotlib

file_path = 'neko.txt.mecab'
words = []

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip() 
        if line == 'EOS':
            continue 

        if '\t' not in line:
            continue

        surface, details = line.split('\t')
        details = details.split(',')
        word = details[6] if len(details) > 6 else surface
        words.append(word)

#出現頻度のカウント
word_counter = Counter(words)

#高い順から並べる
sorted_word_counts = word_counter.most_common(10)

#出現頻度が高い10語の抽出
words, counts = zip(*sorted_word_counts)

#棒グラフを作成
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('単語')
plt.ylabel('出現頻度')
plt.title('出現頻度が高い10語')
plt.xticks(rotation=45)
plt.tight_layout()

#グラフを表示
plt.show()


