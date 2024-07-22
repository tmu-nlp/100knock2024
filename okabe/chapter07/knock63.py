'''
63. 加法構成性によるアナロジー
“Spain”の単語ベクトルから”Madrid”のベクトルを引き，
”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．
'''
from knock60 import model

print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10))

"""
output:
[('Greece', 0.6898480653762817), 
('Aristeidis_Grigoriadis', 0.560684859752655), 
('Ioannis_Drymonakos', 0.5552908778190613), 
('Greeks', 0.545068621635437), 
('Ioannis_Christou', 0.5400863289833069), 
('Hrysopiyi_Devetzi', 0.5248445272445679), 
('Heraklio', 0.5207759737968445), 
('Athens_Greece', 0.516880989074707), 
('Lithuania', 0.5166866183280945), 
('Iraklion', 0.5146791338920593)]
"""