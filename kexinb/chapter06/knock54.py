# task54. 正解率の計測
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ.

import pandas as pd
from sklearn.metrics import accuracy_score
from knock53 import train_pred, test_pred # [max.prob, pred.label]


train = pd.read_csv("output/ch6/train.txt", sep='\t', header=None, 
                    names=['TITLE', 'CATEGORY'])
test = pd.read_csv("output/ch6/test.txt", sep='\t', header=None, 
                   names=['TITLE', 'CATEGORY'])

'''
accuracy_score: fraction of correctly classified samples (float)
(normalize=False -> number of correctly classified samples (int))
'''
train_acc = accuracy_score(train["CATEGORY"], train_pred[1])
test_acc = accuracy_score(test["CATEGORY"], test_pred[1])

if __name__ == "__main__":
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")

# Training Accuracy: 0.919
# Test Accuracy: 0.862