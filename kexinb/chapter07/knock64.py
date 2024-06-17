# task64. アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし，
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し，
# そのベクトルと類似度が最も高い単語と，その類似度を求めよ．
# 求めた単語と類似度は，各事例の末尾に追記せよ

import pickle

if __name__ == "__main__":
    with open("output/ch7/word2vec.pkl", "rb") as f:
        model = pickle.load(f)

    with open("data/questions-words.txt", "r") as txt:
        for line in txt:
            line = line.split()
            if len(line) != 4: # category headers
                print(" ".join(line))
                continue
            word, sim = model.most_similar(positive=[line[1], line[2]], 
                                           negative=[line[0]])[0]
            print(f"{line[0]} {line[1]} {line[2]} {line[3]} | {word} {sim}")

'''
: capital-common-countries
Athens Greece Baghdad Iraq | Iraqi 0.6351870894432068
Athens Greece Bangkok Thailand | Thailand 0.7137669324874878
Athens Greece Beijing China | China 0.7235777378082275
...
: capital-world
: currency
: city-in-state
: family
...
: gram1-adjective-to-adverb
amazing amazingly apparent apparently | apparently 0.48172980546951294
amazing amazingly calm calmly | Calm 0.5576373934745789
amazing amazingly cheerful cheerfully | irrepressibly 0.5931417346000671
...
: gram2-opposite
: gram3-comparative
: gram4-superlative
: gram5-present-participle
: gram6-nationality-adjective
: gram7-past-tense
: gram8-plural
: gram9-plural-verbs
'''