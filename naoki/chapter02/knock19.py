from collections import Counter
import urllib.request
with urllib.request.urlopen("https://nlp100.github.io/data/popular-names.txt") as f:
    name_list = [line.decode('utf-8').split('\t')[0] for line in f]
    name_freq = Counter(name_list)
print(name_freq.most_common())
