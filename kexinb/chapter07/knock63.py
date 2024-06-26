# task63. 加法構成性によるアナロジー
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き，
# ”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．

import pickle

if __name__ == "__main__": 
    with open("output/ch7/word2vec.pkl", "rb") as f:
        model = pickle.load(f)
    # model.most_similar(positive=[], negative=[], 
    #                       topn=10, restrict_vocab=None, indexer=None)
    # model.most_similar(positive=['woman', 'king'], negative=['man'])
    print(model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"]))

'''
[('Greece', 0.6898480653762817), 
('Aristeidis_Grigoriadis', 0.560684859752655), 
('Ioannis_Drymonakos', 0.555290937423706), 
('Greeks', 0.5450686812400818), 
('Ioannis_Christou', 0.5400862693786621), 
('Hrysopiyi_Devetzi', 0.5248445272445679), 
('Heraklio', 0.5207759141921997), 
('Athens_Greece', 0.5168809294700623), 
('Lithuania', 0.5166866183280945), 
('Iraklion', 0.5146791934967041)]
'''
    
