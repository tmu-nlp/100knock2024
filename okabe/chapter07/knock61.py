'''
61. 単語の類似度
“United States”と”U.S.”のコサイン類似度を計算せよ．
'''

from knock60 import model
# similarityでコサイン類似度が求まる
print(model.similarity("United_States", "U.S."))

"""
output:
0.73107743
"""