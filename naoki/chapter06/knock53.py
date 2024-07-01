from sklearn.feature_extraction.text import CountVectorizer
import joblib
# 保存されたモデルを読み込む
LR = joblib.load('model.joblib')

X_test = data_test.iloc[:,1:]
y_test = data_test.iloc[:,0]

y_pred = LR.predict(X_test)