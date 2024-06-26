# 57. Feature weights
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
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

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get feature names
feature_names = X_train.columns

# Get feature weights
feature_weights = model.coef_

# Since we have multiple classes, we will average the absolute values of the weights across all classes
average_weights = np.mean(np.abs(feature_weights), axis=0)

# Create a DataFrame to hold features and their corresponding average weights
feature_importance = pd.DataFrame({'Feature': feature_names, 'Weight': average_weights})

# Sort the DataFrame by weight
feature_importance = feature_importance.sort_values(by='Weight', ascending=False)

# Get the 10 most important features
most_important_features = feature_importance.head(10)

# Get the 10 least important features
least_important_features = feature_importance.tail(10)

print("10 Most Important Features:")
print(most_important_features)

print("\n10 Least Important Features:")
print(least_important_features)
