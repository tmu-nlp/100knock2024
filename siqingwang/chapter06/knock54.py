# 52. Training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Preprocess the data
def preprocess_features(features_df):
    # Convert categorical labels to numerical labels
    le = LabelEncoder()
    features_df['CATEGORY'] = le.fit_transform(features_df['CATEGORY'])
    X = features_df.drop(columns=['CATEGORY', 'TOKENS'])
    y = features_df['CATEGORY']
    return X, y, le

# Load the feature datasets
train_features = pd.read_csv('train.feature.txt', sep='\t')
valid_features = pd.read_csv('valid.feature.txt', sep='\t')
test_features = pd.read_csv('test.feature.txt', sep='\t')

X_train, y_train, label_encoder = preprocess_features(train_features)
X_valid, y_valid, _ = preprocess_features(valid_features)
X_test, y_test, _ = preprocess_features(test_features)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {valid_accuracy}")
print("Validation Classification Report:")
print(classification_report(y_valid, y_valid_pred, target_names=label_encoder.classes_))

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
