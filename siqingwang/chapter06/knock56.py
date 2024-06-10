# 56. Precision, recall and F1 score

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import string

# Download NLTK data files (if not already done)
nltk.download('punkt')


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

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict labels for the training data
y_train_pred = model.predict(X_train)

# Predict labels for the test data
y_test_pred = model.predict(X_test)

# Compute precision, recall, and F1 score for each category
train_report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

# Extract scores for each category
train_scores = pd.DataFrame(train_report).transpose()
test_scores = pd.DataFrame(test_report).transpose()

print("Training Scores:")
print(train_scores)

print("Test Scores:")
print(test_scores)

# Compute micro-average and macro-average for the test data
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_test, y_test_pred, average='micro')
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro')

print(f"Micro-Average Precision: {precision_micro}")
print(f"Micro-Average Recall: {recall_micro}")
print(f"Micro-Average F1 Score: {f1_micro}")

print(f"Macro-Average Precision: {precision_macro}")
print(f"Macro-Average Recall: {recall_macro}")
print(f"Macro-Average F1 Score: {f1_macro}")
