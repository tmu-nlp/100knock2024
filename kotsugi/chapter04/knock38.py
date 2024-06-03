import matplotlib.pyplot as plt
import matplotlib_fontja
from knock35 import count_word

result = count_word()

data = []

for r in result:
  data.append(r[1])

plt.hist(data, bins=20, range=(1, 20))
plt.grid(axis='y')
plt.xlim(xmin=1)
plt.savefig('./kotsugi/chapter04/knock38.png')
