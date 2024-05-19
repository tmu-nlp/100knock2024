#37. 「猫」と共起頻度の高い上位10語
#「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ

import knock30
import matplotlib.pyplot as plt
import japanize_matplotlib

cooccurrence_neko = {}

for sentence in knock30.morph_results:
  if "猫" in [word["surface"] for word in sentence]:
    for i in sentence:
      
      #「猫」自体と記号は共起する語に含めない
      if i["pos"] =="記号":
        continue
      elif i["base"] == "猫":
        continue
      #共起している語が既にcooccurrence_nekoにある場合は頻度を1増やす
      #　　　　　　　　　　　　　　　　　　　　 ない場合は頻度を1にする
      elif i["base"] in cooccurrence_neko:
        cooccurrence_neko[i["base"]]+= 1
      else:
        cooccurrence_neko[i["base"]] = 1

cooccurrence_neko = sorted(cooccurrence_neko.items(), key=lambda x: x[1], reverse=True)

keys = [a[0] for a in cooccurrence_neko[:10]]
values = [a[1] for a in cooccurrence_neko[:10]]

#tick_label：X 軸のラベル
plt.bar(keys, values,  tick_label=keys)
plt.savefig("knock37.png")