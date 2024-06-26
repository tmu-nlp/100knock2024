#３９．Zipfの法則
#単語の出現頻度順位を横軸，その出現頻度を縦軸として，両対数グラフをプロットせよ．

import knock30
import matplotlib.pyplot as plt
import japanize_matplotlib

frequency = {}

for sentence in knock30.morph_results:
  for word in sentence:
    #記号は単語には含めない
    if word["pos"] == "記号":
      pass
    elif word["base"] in frequency:
      frequency[word["base"]]+= 1
    else:
      frequency[word["base"]] = 0

frequency = frequency.values()
frequency = sorted(frequency, key=lambda x: x, reverse=True)
ranks = [r + 1 for r in range(len(frequency))]
plt.scatter(ranks, frequency)
plt.xscale('log')
plt.yscale('log')

plt.xlabel("単語の出現頻度順位")
plt.ylabel("出現頻度")
plt.title("Zipfの法則")

plt.savefig("knock39.png")