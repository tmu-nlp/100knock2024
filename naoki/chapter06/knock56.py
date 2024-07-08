from sklearn.metrics import recall_score, precision_score, f1_score

print(recall_score(y_test, y_pred_test, average='macro'))
print(precision_score(y_test, y_pred_test, average='macro'))
print(f1_score(y_test, y_pred_test, average='macro'))