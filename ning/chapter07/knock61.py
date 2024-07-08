from gensim.models import KeyedVectors

# モデルのロード
model = KeyedVectors.load_word2vec_format("/Users/daining/Desktop/Python/100knock2024/chapter07/GoogleNews-vectors-negative300.bin.gz", binary=True)

# コサイン類似度を計算
similarity = model.similarity("United_States", "U.S.")

print(f"United_StatesとU.S.のコサイン類似度は {similarity}")

#United_StatesとU.S.のコサイン類似度は 0.7310774326324463

"""
自分で計算する場合：
1. 各単語をベクトルに変換
2. これらのベクトル間の内積を計算
3. 各ベクトルの大きさ（ノルム）を計算
4. 内積をノルムで割ることによってコサイン類似度を求める

import numpy as np

# 例として適当に設定
v_US = np.array([0.5, 0.1, 0.3])
v_U.s = np.array([0.4, 0.2, 0.4])

# コサイン類似度の計算
cosine_similarity = np.dot(v_US, v_U.S) / (np.linalg.norm(v_US) * np.linalg.norm(v_U.S))

print(cosine_similarity)
"""