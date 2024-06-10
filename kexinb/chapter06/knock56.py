# task 56. 適合率，再現率，F1スコアの計測
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
# カテゴリごとに適合率，再現率，F1スコアを求め，
# カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ

from sklearn.metrics import classification_report
from knock54 import *

if __name__ == "__main__":
    print(classification_report(test["CATEGORY"], test_pred[1]))

'''
Training Accuracy: 0.919
Test Accuracy: 0.862
              precision    recall  f1-score   support

           b       0.85      0.93      0.89       558
           e       0.88      0.95      0.92       522
           m       0.88      0.49      0.63        90
           t       0.79      0.54      0.64       164

    accuracy                           0.86      1334
   macro avg       0.85      0.73      0.77      1334
weighted avg       0.86      0.86      0.85      1334
'''
