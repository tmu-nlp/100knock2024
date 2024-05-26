#38. ヒストグラム
#単語の出現頻度のヒストグラムを描け．
# ただし，横軸は出現頻度を表し，1から単語の出現頻度の最大値までの線形目盛とする．縦軸はx軸で示される出現頻度となった単語の異なり数（種類数）である．

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
#単語の異なり数を格納
frequency = frequency.values()
frequency = sorted(frequency, key=lambda x: x, reverse=True)

plt.hist(frequency, bins=50)
plt.xlabel("単語の異なり数")
plt.ylabel("出現頻度")
plt.title("単語の出現頻度のヒストグラム")
plt.savefig("knock38.png")