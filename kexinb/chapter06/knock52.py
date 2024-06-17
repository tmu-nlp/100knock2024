#task 52. 学習
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Initialize logistic regression model
lr = LogisticRegression(random_state=42, max_iter=1000)

# Load training data with headers
X_train = pd.read_csv("output/ch6/train.feature.txt", sep='\t')
Y_train = pd.read_csv("output/ch6/train.txt", sep='\t', header=None, 
                      names=['TITLE', 'CATEGORY'])['CATEGORY']

# Train the model
lr.fit(X_train, Y_train)

# Save the trained model
with open("output/ch6/logreg.pkl", "wb") as f:
    pickle.dump(lr, f)

