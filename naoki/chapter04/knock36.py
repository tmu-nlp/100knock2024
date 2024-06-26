import matplotlib.pyplot as plt
#import japanize_matplotlib
import collections
%matplotlib inline
word_list_top10 = []
word_list_count = []
for i in range(10):
    word_list_top10.append(word_list_rank[:10][i][0])
    word_list_count.append(word_list_rank[:10][i][1])
plt.bar(x = word_list_top10,height = word_list_count)

