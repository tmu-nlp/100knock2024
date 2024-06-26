from knock34 import get_ma_list_for_sentence
def calc_cooccurrence(word):
  result = {}
  ma_list = get_ma_list_for_sentence()

  for sentence in ma_list:
    for i, d in enumerate(sentence):
      if (d["surface"] == word):
        for s in sentence:
          if (s["pos"] == "補助記号"):
            continue
          if (s["surface"] != word):
            if (s['base'] in result):
              result[f"{s['base']}"] += 1
            else:
              result[f"{s['base']}"] = 1
  return sorted(result.items(), reverse=True, key=lambda x:x[1])

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib_fontja

  data = calc_cooccurrence('猫')[0:10]
  print(data)

  plt.bar(*zip(*data))
  plt.savefig('./kotsugi/chapter04/knock37.png')
  
