'''
74. 正解率の計測Permalink
問題73で求めた行列を用いて学習データおよび評価データの事例を分類したとき，その正解率をそれぞれ求めよ．
'''
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from load_vector_data import *

#model
net = torch.nn.Linear(300, 4, bias=False)
net.load_state_dict(torch.load("model.pt"))

#accuracy
#torch.maxはtensorの(最大値, 最大値のインデックス)を返す
y_max_train, y_pred_train = torch.max(net(x_train),dim=1)
print(f"train data acc:{accuracy_score(y_pred_train, y_train)}")
y_max_test, y_pred_test = torch.max(net(x_test),dim=1)
print(f"test data acc:{accuracy_score(y_pred_test, y_test)}")