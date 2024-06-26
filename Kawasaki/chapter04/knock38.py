import knock35
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import OrderedDict

freq = list(knock35.word_dict.values())

plt.hist(freq,bins=10)
plt.xlabel("単語の異なり度数")
plt.ylabel("出現頻度")
plt.title("単語の出現頻度のヒストグラム")
#plt.savefig("knock38.png")