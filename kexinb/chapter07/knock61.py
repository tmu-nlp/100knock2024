# task61. 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ．

import pickle

with open("output/ch7/word2vec.pkl", "rb") as f:
    model = pickle.load(f)

if __name__ == "__main__":
    print(model.similarity("United_States", "U.S."))
    # 0.73107743
