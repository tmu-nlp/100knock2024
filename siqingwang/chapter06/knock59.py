import knock58
from knock58 import X_train
from knock58 import X_valid
from knock58 import X_test
from knock58 import train
from knock58 import valid
from knock58 import test

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np

c = {'C': [0.01, 0.05, 0.1]}

gs_model = GridSearchCV(LogisticRegression(max_iter=1500), c, cv=5, verbose=1)
gs_model.fit(X_train,train['CATEGORY'])

#best parameter
gs_model1 = gs_model.best_estimator_
print(format(gs_model1.score(X_train, train['CATEGORY'])))
print(format(gs_model1.score(X_valid, valid['CATEGORY'])))
print(format(gs_model1.score(X_test, test['CATEGORY'])))