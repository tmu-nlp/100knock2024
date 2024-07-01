from sklearn.metrics import confusion_matrix
train_con = confusion_matrix(y_train, y_pred_train)
test_con = confusion_matrix(y_test, y_pred_test)
print('訓練データの混合行列')
print(train_con)
print('テストデータの混合行列')
print(test_con)