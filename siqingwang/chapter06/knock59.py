# 59. Hyper-parameter tuning

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# Preprocess the data
def preprocess_features(features):
    le = LabelEncoder()
    features['CATEGORY'] = le.fit_transform(features['CATEGORY'])
    X = features.drop(columns=['CATEGORY', 'TOKENS'])
    y = features['CATEGORY']
    return X, y, le


train_features = pd.read_csv('train.feature.txt', sep='\t')
valid_features = pd.read_csv('valid.feature.txt', sep='\t')
test_features = pd.read_csv('test.feature.txt', sep='\t')

X_train, y_train, label_encoder = preprocess_features(train_features)
X_valid, y_valid, _ = preprocess_features(valid_features)
X_test, y_test, _ = preprocess_features(test_features)

# Combine train and validation datasets for cross-validation during GridSearch
X_combined = pd.concat([X_train, X_valid])
y_combined = np.concatenate([y_train, y_valid])

# Define a grid of hyperparameters
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs'],
    'penalty': ['l2']
}

# Initialize the logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)

# Perform grid search with cross-validation
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_combined, y_combined)

# Get the best model
best_model = grid_search.best_estimator_

# Print the best parameters and best score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated accuracy: {grid_search.best_score_}")

# Evaluate the best model on the test data
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Test accuracy: {test_accuracy}")
