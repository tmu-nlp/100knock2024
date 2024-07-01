from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.linear_model import LogisticRegression

X_train = data_train.iloc[:,1:]
y_train = data_train.iloc[:,0]

LR = LogisticRegression(penalty='l1', solver='saga', random_state=777)
LR.fit(X_train, y_train)
y_pred_train = LR.predict(X_train)
joblib.dump(LR, 'model.joblib')