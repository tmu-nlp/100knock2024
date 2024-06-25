#70.単語ベクトルの和による特徴量

import torch
from gensim.models import KeyedVectors
import numpy as np
import re

# 単語ベクトルの読み込み
print("Loading word vectors...")
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def load_and_vectorize(filename):
    data = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            category = parts[0]
            title = ' '.join(parts[1:])  # タイトルに含まれるタブを空白に置換

            # カテゴリをラベルに変換
            if category == 'b':
                label = 0
            elif category == 't':
                label = 1
            elif category == 'e':
                label = 2
            elif category == 'm':
                label = 3
            else:
                print(f"Skipping unknown category: {category}")
                continue

            # タイトルを単語に分割
            words = re.findall(r'\w+', title.lower())
            
            # 単語ベクトルの平均を計算
            vec = np.zeros(300)
            count = 0
            for word in words:
                if word in word_vectors:
                    vec += word_vectors[word]
                    count += 1
            if count > 0:
                vec /= count
            else:
                print(f"Skipping title with no valid words: {title}")
                continue
            
            data.append(vec)
            labels.append(label)
    
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# データの読み込みと特徴量計算
print("Processing train data...")
Xtrain, Ytrain = load_and_vectorize('train.txt')
print("Processing valid data...")
Xvalid, Yvalid = load_and_vectorize('valid.txt')
print("Processing test data...")
Xtest, Ytest = load_and_vectorize('test.txt')

# データの保存
print("Saving data...")
torch.save(Xtrain, 'Xtrain.pt')
torch.save(Ytrain, 'Ytrain.pt')
torch.save(Xvalid, 'Xvalid.pt')
torch.save(Yvalid, 'Yvalid.pt')
torch.save(Xtest, 'Xtest.pt')
torch.save(Ytest, 'Ytest.pt')

print("Data processing and saving completed.")

# データの形状を表示
print(f"\nXtrain shape: {Xtrain.shape}")
print(f"Ytrain shape: {Ytrain.shape}")
print(f"\nXvalid shape: {Xvalid.shape}")
print(f"Yvalid shape: {Yvalid.shape}")
print(f"\nXtest shape: {Xtest.shape}")
print(f"Ytest shape: {Ytest.shape}")
