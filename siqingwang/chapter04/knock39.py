
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter
from morphological_analysis import morphology_map

file_parsed = "./neko.txt.mecab"
words = morphology_map(file_parsed)
words_without_punctuation = []
for word in words:
    if word['pos'] != '記号':
        words_without_punctuation.append(word)

word_count = Counter()
word_count.update([word['surface'] for word in words_without_punctuation])

items  = word_count.most_common()
items  = list(zip(*items))
counts = items[1]
rank = list(range(1, len(counts) + 1))

fig = plt.figure()
plt.scatter(counts, rank, alpha=0.6)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.xscale('log')
plt.yscale('log')
plt.title('Zipf\'s law')
plt.show()