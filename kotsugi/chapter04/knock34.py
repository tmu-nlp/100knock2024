from knock30 import get_ma_list

def get_ma_list_for_sentence():
  result = get_ma_list()
  parent = []
  child = []
  for r in result:
    child.append(r)
    if (r["pos1"] == '句点' or r["pos"] == '空白'):
      parent.append(child)
      child = []

  return parent

if __name__ == "__main__":
  result = get_ma_list_for_sentence()

  noun = ''
  max_noun_size = 0

  for sentence in result:
    noun_size = 0
    now_noun = ''
    is_noun = False

    for target in sentence:
      if (target["pos"] == "名詞"):
        noun_size += 1
        now_noun += target["surface"]
        is_noun = True
      
      elif (is_noun):
        if (noun_size > max_noun_size):
          max_noun_size = noun_size
          noun = now_noun
        
        noun_size = 0
        now_noun = ''
        is_noun = False
  
  print(noun, max_noun_size)
  # 明治三十八年何月何日戸締り 9
