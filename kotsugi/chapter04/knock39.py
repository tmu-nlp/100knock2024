import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_fontja
from knock35 import count_word

result = count_word()

data = []

for r in result:
  data.append(r[1])

data = {'value': data}

df = pd.DataFrame(data)
df['ranks'] = df.rank(ascending=False)

rank_order = df['ranks'].argsort()
ordered_data = df.iloc[rank_order]

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.scatter(ordered_data.ranks.tolist(), ordered_data.value.tolist())
plt.savefig('./kotsugi/chapter04/knock39.png')
