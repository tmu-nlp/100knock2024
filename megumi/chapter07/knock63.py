#63. 加法構成性によるアナロジー
#“Spain”の単語ベクトルから”Madrid”のベクトルを引き，
# ”Athens”のベクトルを足したベクトルを計算し，
# そのベクトルと類似度の高い10語とその類似度を出力せよ．

#自然言語処理用のgensimパッケージ
from gensim.models import KeyedVectors
file = './GoogleNews-vectors-negative300.bin.gz'

#fileをword2vec形式で読み込み
model = KeyedVectors.load_word2vec_format(file,binary = True)

res = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
for i, x in enumerate(res):
    print('{}\t{}\t{}'.format(i + 1, x[0], x[1]))

"""
出力結果
1       Greece  0.6898480653762817
2       Aristeidis_Grigoriadis  0.5606847405433655
3       Ioannis_Drymonakos      0.5552908778190613
4       Greeks  0.5450685620307922
5       Ioannis_Christou        0.5400863289833069
6       Hrysopiyi_Devetzi       0.5248444676399231
7       Heraklio        0.5207759737968445
8       Athens_Greece   0.516880989074707
9       Lithuania       0.5166866183280945
10      Iraklion        0.5146791338920593
"""