from gensim.models import KeyedVectors

# モデルのロード
model = KeyedVectors.load_word2vec_format("/Users/daining/Desktop/Python/100knock2024/chapter07/GoogleNews-vectors-negative300.bin.gz", binary=True)

# アナロジーベクトルを計算し、類似度の高い10語を取得
similar_words = model.most_similar_cosmul(positive=["Spain", "Athens"], negative=["Madrid"], topn=10)

# 結果を出力
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

"""
Greece: 0.9562304615974426
Aristeidis_Grigoriadis: 0.8694582581520081
Ioannis_Drymonakos: 0.8600283265113831
Ioannis_Christou: 0.8544449806213379
Greeks: 0.8521003127098083
Hrysopiyi_Devetzi: 0.8383886814117432
Panagiotis_Gionis: 0.8323913216590881
Heraklio: 0.8297829627990723
Lithuania: 0.8291547298431396
Periklis_Iakovakis: 0.8289120197296143
"""