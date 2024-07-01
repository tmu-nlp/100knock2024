#62. 類似度の高い単語10件
#“United States”とコサイン類似度が高い10語と，
# その類似度を出力せよ．

#自然言語処理用のgensimパッケージ
from gensim.models import KeyedVectors
file = './GoogleNews-vectors-negative300.bin.gz'

#fileをword2vec形式で読み込み
model = KeyedVectors.load_word2vec_format(file,binary = True)

res = model.most_similar('United_States', topn=10)
for i, x in enumerate(res):
    print('{}\t{}\t{}'.format(i + 1, x[0], x[1]))

"""
出力結果
1       Unites_States   0.7877249717712402
2       Untied_States   0.7541371583938599
3       United_Sates    0.74007248878479
4       U.S.    0.7310774326324463
5       theUnited_States        0.6404393911361694
6       America 0.6178409457206726
7       UnitedStates    0.6167312264442444
8       Europe  0.6132988929748535
9       countries       0.6044804453849792
10      Canada  0.6019068956375122
"""