# task 55. 混同行列の作成
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を
# 学習データおよび評価データ上で作成せよ

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from knock53 import train_pred, test_pred


train = pd.read_csv("output/ch6/train.txt", sep='\t', header=None, 
                    names=['TITLE', 'CATEGORY'])
test = pd.read_csv("output/ch6/test.txt", sep='\t', header=None, 
                   names=['TITLE', 'CATEGORY'])

train_con = confusion_matrix(train["CATEGORY"], train_pred[1])
test_con = confusion_matrix(test["CATEGORY"], test_pred[1])

if __name__ == "__main__":
    print("Confusion Matrix (Train)")
    print(train_con)
    print("Confusion Matrix (Test)")
    print(test_con)

'''
Confusion Matrix (Train)
[[4368  101    9   60]
 [  64 4153    2    9]
 [  95  140  455   11]
 [ 208  154    8  835]]

Confusion Matrix (Test)
[[521  18   4  15]
 [ 19 497   0   6]
 [ 24  20  44   2]
 [ 46  28   2  88]]
'''

sns.heatmap(train_con, annot=True, cmap="Greens")
plt.savefig("output/ch6/train_confusion_matrix.png")
plt.clf()
sns.heatmap(test_con, annot=True, cmap="Greens")
plt.savefig("output/ch6/test_confusion_matrix.png")