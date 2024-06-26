from knock30 import get_ma_list

result = get_ma_list()

for r in result:
  if (r["pos"] == "動詞"):
    print(r["base"])
