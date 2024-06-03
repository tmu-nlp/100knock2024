from knock30 import get_ma_list

result = get_ma_list()

for i in range(len(result) - 2):
  target = result[i]
  no_target = result[i + 1]
  next_target = result[i + 2]
  if (target["pos"] == "名詞" and no_target["surface"] == "の" and next_target["pos"] == "名詞"):
    print(f"{target["surface"]}の{next_target["surface"]}")
