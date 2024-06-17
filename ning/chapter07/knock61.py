from gensim.models import KeyedVectors

# モデルのロード
model = KeyedVectors.load_word2vec_format("/Users/daining/Desktop/Python/100knock2024/chapter07/GoogleNews-vectors-negative300.bin.gz", binary=True)

# コサイン類似度を計算
similarity = model.similarity("United_States", "U.S.")

print(f"United_StatesとU.S.のコサイン類似度は {similarity}")

#United_StatesとU.S.のコサイン類似度は 0.7310774326324463