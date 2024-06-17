#63. 加法構成性によるアナロジー
#“Spain”の単語ベクトルから”Madrid”のベクトルを引き，
# ”Athens”のベクトルを足したベクトルを計算し，
# そのベクトルと類似度の高い10語とその類似度を出力せよ．

res = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
for i, x in enumerate(res):
    print('{}\t{}\t{}'.format(i + 1, x[0], x[1]))

"""
1	Greece	0.6898480653762817
2	Aristeidis_Grigoriadis	0.560684859752655
3	Ioannis_Drymonakos	0.5552908778190613
4	Greeks	0.545068621635437
5	Ioannis_Christou	0.5400862097740173
6	Hrysopiyi_Devetzi	0.5248445272445679
7	Heraklio	0.5207759737968445
8	Athens_Greece	0.516880989074707
9	Lithuania	0.5166865587234497
10	Iraklion	0.5146791338920593
"""