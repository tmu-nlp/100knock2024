from knock35 import sort_word_dict
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import OrderedDict

x = np.arange(len(sort_word_dict)) + [1] * len(sort_word_dict)
# print(x)

plt.scatter(x, sort_word_dict.values())
plt.xscale("log")
plt.yscale("log")
plt.xlabel("単語の出現頻度順位")
plt.ylabel("出現頻度")
plt.title("Zipfの法則")
plt.savefig("knock39.png")