from collections import Counter

file_path = 'neko.txt.mecab'
output_file_path = 'word_frequencies.txt'
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
        word = details[6] if len(details) > 6 else surface  #基本形が存在すれば基本形を、なければ表層形を
        words.append(word)

#単語の出現頻度をカウント
word_counter = Counter(words)

#出現頻度の高い順から並べる
sorted_word_counts = word_counter.most_common()

#ファイルに出力
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for word, count in sorted_word_counts:
        out_file.write(f"{word}\t{count}\n")

