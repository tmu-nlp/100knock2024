#61. 単語の類似度
#“United States”と”U.S.”のコサイン類似度を計算せよ．

#自然言語処理用のgensimパッケージ
from gensim.models import KeyedVectors
file = './GoogleNews-vectors-negative300.bin.gz'

#fileをword2vec形式で読み込み
model = KeyedVectors.load_word2vec_format(file,binary = True)

#simirarityメソッドで単語の類似度を取得
print(model.similarity('United_States','U.S.'))

"""
出力結果
0.7310774
"""
