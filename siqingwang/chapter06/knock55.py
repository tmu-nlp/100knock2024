# 55. Confusion matrix

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Predict labels for the training data
y_train_pred = model.predict(X_train)

# Predict labels for the test data
y_test_pred = model.predict(X_test)

# Generate confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)

# Generate confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot confusion matrix for training data
plot_confusion_matrix(conf_matrix_train, label_encoder.classes_, title='Confusion Matrix - Training Data')

# Plot confusion matrix for test data
plot_confusion_matrix(conf_matrix_test, label_encoder.classes_, title='Confusion Matrix - Test Data')
