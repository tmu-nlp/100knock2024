import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Step 1: reading file
data = pd.read_csv('newsCorpora.csv', sep = '\t', header = None, names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
publishers = ['Reuters', 'Huffington Post', 'Businessweek', '“Contactmusic.com', 'Daily Mail']
data = data[data['PUBLISHER'].isin(publishers)]
data = data[['TITLE', 'CATEGORY']]

# Shuffle
train, valid_test = train_test_split(data, test_size = 0.2, random_state = 0, shuffle = True, stratify = data['CATEGORY'])
valid, test = train_test_split(valid_test, test_size=0.5, random_state=0, shuffle = True, stratify=valid_test['CATEGORY'])

vectorizer = CountVectorizer()
# Fit the vectorizer on the training data headlines and transform the data
X_train = vectorizer.fit_transform(train['TITLE'])
X_valid = vectorizer.transform(valid['TITLE'])
X_test = vectorizer.transform(test['TITLE'])

# Get the category labels
y_train = train['CATEGORY']
y_valid = valid['CATEGORY']
y_test = test['CATEGORY']

# Step 2: Train the logistic regression model with different regularization parameters
C_values = np.logspace(-4, 4, 10)
train_accuracies = []
valid_accuracies = []
test_accuracies = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict the categories for training, validation, and test data
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Compute the accuracy scores
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    valid_accuracies.append(accuracy_score(y_valid, y_valid_pred))
    test_accuracies.append(accuracy_score(y_test, y_test_pred))

# Step 3: Plot the accuracy scores
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(C_values, valid_accuracies, label='Validation Accuracy', marker='o')
plt.plot(C_values, test_accuracies, label='Test Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy Score')
plt.title('Effect of Regularization Parameter on Accuracy')
plt.legend()
plt.grid(True)
plt.show()
