import matplotlib.pyplot as plt
word_list = []
for sentense in morphemes:
    for text in sentense:
        word_list.append(text['surface'])
hist = collections.Counter(word_list)
plt.hist(hist.values(),range(1,30))
