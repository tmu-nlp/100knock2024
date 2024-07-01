# task74. 正解率の計測

from knock73 import *


acc_train = calc_acc(model, dataloader_train)
acc_test = calc_acc(model, dataloader_test)

print(f"train_acc : {acc_train}")
print(f"test_acc : {acc_test}")

"""
train_acc : 0.9257801518133258
test_acc : 0.8964741185296324
"""