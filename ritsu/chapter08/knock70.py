import pandas as pd
import pickle
import string
import torch
import os
from gensim.models import KeyedVectors

def trans_word2vec(text, model):
    table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # 記号を空白に
    words = text.translate(table).split()  # スペースで分割
    vec = [model[word] for word in words if word in model]  # 1語ずつベクトル化
    return torch.tensor(sum(vec) / len(vec)) if vec else torch.zeros(model.vector_size)  # 平均ベクトルをTensor型に変換して出力

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 単語ベクトルの読み込み
    word_vectors_path = os.path.join(current_dir, 'GoogleNews-vectors-negative300.bin')
    model = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

    # データの読み込み
    train = pd.read_csv(os.path.join(os.path.dirname(current_dir), 'chapter06', "train.txt"), sep='\t', names=['CATEGORY', 'TITLE'])
    valid = pd.read_csv(os.path.join(os.path.dirname(current_dir), 'chapter06', "valid.txt"), sep='\t', names=['CATEGORY', 'TITLE'])
    test = pd.read_csv(os.path.join(os.path.dirname(current_dir), 'chapter06', "test.txt"), sep='\t', names=['CATEGORY', 'TITLE'])

    # 特徴量ベクトルの作成
    X_train = torch.stack([trans_word2vec(text, model) for text in train['TITLE']])
    X_valid = torch.stack([trans_word2vec(text, model) for text in valid['TITLE']])
    X_test = torch.stack([trans_word2vec(text, model) for text in test['TITLE']])

    # カテゴリのエンコーディング
    category = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    y_train = torch.tensor(train['CATEGORY'].map(lambda x: category[x]).values)
    y_valid = torch.tensor(valid['CATEGORY'].map(lambda x: category[x]).values)
    y_test = torch.tensor(test['CATEGORY'].map(lambda x: category[x]).values)

    # データの保存
    torch.save(X_train, os.path.join(current_dir, 'X_train.pt'))
    torch.save(X_valid, os.path.join(current_dir, 'X_valid.pt'))
    torch.save(X_test, os.path.join(current_dir, 'X_test.pt'))
    torch.save(y_train, os.path.join(current_dir, 'y_train.pt'))
    torch.save(y_valid, os.path.join(current_dir, 'y_valid.pt'))
    torch.save(y_test, os.path.join(current_dir, 'y_test.pt'))

    print(f"X_train shape: {X_train.shape}")
    print(f"X_train : {X_train}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"X_valid : {X_valid}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test : {X_test}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train : {y_train}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"y_valid : {y_valid}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_test : {y_test}")
    print("Data processing and saving completed.")

if __name__ == "__main__":
    main()

"""
X_train shape: torch.Size([10672, 300])
X_train : tensor([[ 0.0286, -0.0328, -0.0558,  ..., -0.0555,  0.0478,  0.0240],
        [-0.0124,  0.0470, -0.0338,  ...,  0.0683,  0.0964,  0.0034],
        [-0.0840,  0.0300, -0.0437,  ..., -0.1123,  0.0691,  0.0279],
        ...,
        [ 0.1067,  0.1114, -0.0505,  ...,  0.0473,  0.0949, -0.0323],
        [ 0.0689, -0.0176,  0.0098,  ..., -0.0861,  0.0655, -0.0741],
        [ 0.0681,  0.0661,  0.0223,  ...,  0.0635,  0.0737,  0.0103]])
X_valid shape: torch.Size([1334, 300])
X_valid : tensor([[-0.0573,  0.0061, -0.0600,  ...,  0.0899,  0.0736, -0.0577],
        [-0.0466,  0.0254,  0.0426,  ...,  0.0809,  0.0119, -0.0096],
        [ 0.0623,  0.0577, -0.0665,  ...,  0.0273,  0.1403,  0.0122],
        ...,
        [ 0.0504,  0.0028, -0.1159,  ..., -0.0466,  0.0029,  0.0064],
        [-0.0278,  0.0665,  0.0640,  ...,  0.0435,  0.1157,  0.0702],
        [ 0.0466,  0.1059,  0.1113,  ..., -0.0032,  0.0954, -0.0244]])
X_test shape: torch.Size([1334, 300])
X_test : tensor([[ 0.0641,  0.0523, -0.0404,  ..., -0.1107, -0.0762,  0.0451],
        [ 0.0437,  0.0108,  0.0270,  ..., -0.0053,  0.0644,  0.0060],
        [ 0.0244,  0.1348, -0.0008,  ..., -0.0335,  0.0303,  0.0072],
        ...,
        [ 0.0710,  0.1102, -0.1216,  ...,  0.1396,  0.0369, -0.0077],
        [ 0.1001, -0.0236, -0.0225,  ..., -0.0151,  0.0235,  0.0121],
        [-0.0066,  0.0729, -0.0280,  ..., -0.0181,  0.0407, -0.0574]])
y_train shape: torch.Size([10672])
y_train : tensor([2, 0, 1,  ..., 2, 0, 0])
y_valid shape: torch.Size([1334])
y_valid : tensor([0, 0, 0,  ..., 2, 0, 0])
y_test shape: torch.Size([1334])
y_test : tensor([1, 2, 2,  ..., 0, 0, 2])
Data processing and saving completed.
"""