from knock35 import word_dict
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import OrderedDict

top10 = OrderedDict(sorted(word_dict.items(), key = lambda x:x[1], reverse=True)[:10]) 


label = []
data = []

for k,v in top10.items():
    label.append(k)
    data.append(v)
x = [1,2,3,4,5,6,7,8,9,10]
plt.bar(x,data)
plt.xticks(x,label)
plt.xlabel("語")
plt.ylabel("出現頻度")
plt.title("頻度上位10語")
#plt.savefig("knock36.png")