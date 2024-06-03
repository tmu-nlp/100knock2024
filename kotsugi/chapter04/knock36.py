import matplotlib.pyplot as plt
import matplotlib_fontja
from knock35 import count_word

data = count_word()[0:10]

plt.bar(*zip(*data))
plt.savefig('./kotsugi/chapter04/knock36.png')
