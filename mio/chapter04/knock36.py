#頻度上位10語
#出現頻度が高い10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

import knock35
import matplotlib.pyplot as plt
import japanize_matplotlib


keys = [a[0] for a in knock35.frequency[:10]]
values = [a[1] for a in knock35.frequency[:10]]

plt.bar(keys, values)
plt.savefig("knock36_2.png")