#60. 単語ベクトルの読み込みと表示
#Google Newsデータセット（約1,000億単語）での
# 学習済み単語ベクトル（300万単語・フレーズ，300次元）をダウンロードし，
# ”United States”の単語ベクトルを表示せよ．
# ただし，”United States”は内部的には”United_States”と表現されていることに注意せよ．

from gensim.models import KeyedVectors

# モデルファイルのパス
file = './GoogleNews-vectors-negative300.bin.gz'

# モデルの読み込み
model = KeyedVectors.load_word2vec_format(file, binary=True)

# "United States"の単語ベクトルを表示
print(model['United_States'])

"""
[-3.61328125e-02 -4.83398438e-02  2.35351562e-01  1.74804688e-01
 -1.46484375e-01 -7.42187500e-02 -1.01562500e-01 -7.71484375e-02
  1.09375000e-01 
"""