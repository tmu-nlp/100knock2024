# task58. 正則化パラメータの変更
# 異なる正則化パラメータでロジスティック回帰モデルを学習し，
# 学習データ，検証データ，および評価データ上の正解率を求めよ．
# 実験の結果は，正則化パラメータを横軸，正解率を縦軸としたグラフにまとめよ．

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X_train = pd.read_table("output/ch6/train.feature.txt")
    Y_train = pd.read_csv("output/ch6/train.txt", sep='\t', header=None, 
                          names=['TITLE', 'CATEGORY'])['CATEGORY']

    X_valid = pd.read_table("output/ch6/valid.feature.txt")
    Y_valid = pd.read_csv("output/ch6/valid.txt", sep='\t', header=None, 
                          names=['TITLE', 'CATEGORY'])['CATEGORY']

    X_test = pd.read_table("output/ch6/test.feature.txt")
    Y_test = pd.read_csv("output/ch6/test.txt", sep='\t', header=None, 
                          names=['TITLE', 'CATEGORY'])['CATEGORY']

    C_list = [1e-2, 1e-1, 1.0, 1e+1, 1e+2]
    
    train_acc = []
    valid_acc = []
    test_acc = []

    for C in C_list:
        lr = LogisticRegression(random_state=0, max_iter=1000, C=C)
        lr.fit(X_train, Y_train)

        with open(f"output/ch6/logreg_{C}.pkl", "wb") as f:
            pickle.dump(lr, f)

        train_acc.append(accuracy_score(Y_train, lr.predict(X_train)))
        valid_acc.append(accuracy_score(Y_valid, lr.predict(X_valid)))
        test_acc.append(accuracy_score(Y_test, lr.predict(X_test)))


    plt.plot(C_list, train_acc, label="train")
    plt.plot(C_list, valid_acc, label="valid")
    plt.plot(C_list, test_acc, label="test")
    plt.xlabel("Regularization Parameter C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xscale("log")
    plt.savefig("output/ch6/output58.png", format="png")