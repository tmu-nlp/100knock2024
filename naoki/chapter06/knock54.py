from sklearn.metrics import accuracy_score
LR = joblib.load('model.joblib')

X_test = data_test.iloc[:,1:]
y_test = data_test.iloc[:,0]

y_pred_test = LR.predict(X_test)
accuracy_score(y_test, y_pred_test)