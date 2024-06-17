# task62. 類似度の高い単語10件
# “United States”とコサイン類似度が高い10語と，その類似度を出力せよ．

import pickle

if __name__ == "__main__": 
    with open("output/ch7/word2vec.pkl", "rb") as f:
        model = pickle.load(f)
    # Word2Vec.most_similar(positive=[], negative=[], 
    #                       topn=10, restrict_vocab=None, indexer=None)
    print(model.most_similar("United_States"))

'''
[('Unites_States', 0.7877248525619507), 
('Untied_States', 0.7541370391845703), 
('United_Sates', 0.7400725483894348), 
('U.S.', 0.7310774326324463), 
('theUnited_States', 0.6404393911361694), 
('America', 0.6178410053253174), 
('UnitedStates', 0.6167312264442444), 
('Europe', 0.6132988929748535), 
('countries', 0.6044804453849792), 
('Canada', 0.6019070148468018)]
'''
