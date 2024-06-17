'''
64. アナロジーデータでの実験
単語アナロジーの評価データをダウンロードし，vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
そのベクトルと類似度が最も高い単語と，その類似度を求めよ．求めた単語と類似度は，各事例の末尾に追記せよ．
'''

from knock60 import model

with open('questions-words.txt', 'r') as input, open('questions-words-add.txt', 'w') as output:
  for line in input:  # inputから1行ずつ読込み、求めた単語と類似度を追加してf2に書込む
    line = line.split()
    if line[0] == ':':
      category = line[1]
      print(category)
    else:
      word, cos = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)[0]
      output.write(' '.join([category] + line + [word, str(cos) + '\n']))

"""
output-file:
    question-words-add.txt
    
category:
    capital-common-countries
    capital-world
    currency
    city-in-state
    family
    gram1-adjective-to-adverb
    gram2-opposite
    gram3-comparative
    gram4-superlative
    gram5-present-participle
    gram6-nationality-adjective
    gram7-past-tense
    gram8-plural
    gram9-plural-verbs
"""