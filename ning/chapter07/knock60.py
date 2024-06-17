from gensim.models import KeyedVectors

# モデルのロード
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

# "United_States"の単語ベクトルを取得
vector = model.get_vector("United_States")

# ベクトルをファイルに保存
with open("United_States_vector.txt", "w") as f:
    for value in vector:
        f.write(f"{value}\n")

print("ファイルが保存されました")
