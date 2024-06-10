def get_ma_list():
  lst = []
  with open('./kotsugi/chapter04/neko.txt.mecab', 'r') as f: 
    for line in f:
      if ('EOS' in line): break

      surface = line.split('\t')[0]
      features = line.split('\t')[1].split(',')
      pos = features[0]
      pos1 = features[1]
      base = surface
      if (len(features) >= 11):
        base = features[10]

      lst.append({
        "surface": surface,
        "base": base,
        "pos": pos,
        "pos1": pos1,
      })
  return lst

if __name__ == "__main__":
  import MeCab

  # 辞書：unidic
  tagger = MeCab.Tagger()

  with open('./kotsugi/chapter04/neko.txt', 'r', encoding='utf-8') as f:
    text = f.read()

  with open('./kotsugi/chapter04/neko.txt.mecab', 'w+', encoding='utf-8') as wf:
    wf.write(tagger.parse(text))
  
  result = get_ma_list()

  for i in range(10):
    print(result[i])
