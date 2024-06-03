from knock34 import get_ma_list

def count_word():
  result = get_ma_list()
  dct = {}

  for r in result:
    if (r["pos"] == "補助記号"):
      continue
    if (r['base'] in dct):
      dct[f"{r['base']}"] += 1
    else:
      dct[f"{r['base']}"] = 1

  return sorted(dct.items(), reverse=True, key=lambda x:x[1])

if __name__ == "__main__":
  print(count_word()[0:10])