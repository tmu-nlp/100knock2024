import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

para = [0.01 , 0.1 , 1 , 10]

X_train = data_train.iloc[:,1:]
y_train = data_train.iloc[:,0]

X_valid = data_valid.iloc[:,1:]
y_valid = data_valid.iloc[:,0]

X_test = data_test.iloc[:,1:]
y_test = data_test.iloc[:,0]

train_accuracy = []
valid_accuracy = []
test_accuracy = []

#appendするリストを間違えてエラーを起こしてしまった。折角学習していた時間がもったいない。解決策はないか

for c in para:
    LR = LogisticRegression(penalty='l1', C=c, solver='saga', random_state=777)
    LR.fit(X_train, y_train)
    #train
    y_pred_train = LR.predict(X_train)
    train_accuracy.append(accuracy_score(y_train, y_pred_train))
    #valid
    y_pred_valid = LR.predict(X_valid)
    valid_accuracy.append(accuracy_score(y_valid, y_pred_valid))
    #test
    y_pred_test = LR.predict(X_test)
    test_accuracy.append(accuracy_score(y_test, y_pred_test))


plt.plot(para, train_accuracy, label='train')
plt.plot(para, valid_accuracy, label='valid')
plt.plot(para, test_accuracy, label='test')
plt.legend()
plt.show()



    