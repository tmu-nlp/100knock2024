import itertools
import matplotlib.pyplot as plt
%matplotlib inline

related_list = []
for sentense in morphemes:
    for i in range(len(sentense)-1):
        if sentense[i]['surface'] == '猫' and sentense[i+1]["pos"] != "補助記号" and sentense[i+1]["pos"] != '助詞' and sentense[i+1]["pos"] != '助動詞':
            related_list.append(sentense[i+1]['surface'])
all_neko = list(itertools.chain.from_iterable(related_list))
count_list = collections.Counter(all_neko)
word_list = []
height_list = []
print(count_list)
for i in range(10):
    word_list.append(count_list.most_common()[:10][i][0])
    height_list.append(count_list.most_common()[:10][i][1])
plt.bar(x = word_list, height = height_list)
