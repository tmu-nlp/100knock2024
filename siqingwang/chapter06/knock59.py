import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load data
train_data = pd.read_csv('train.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])
test_data = pd.read_csv('test.txt', sep='\t', header=None, names=['CATEGORY', 'TITLE'])

X_train, y_train = train_data['TITLE'], train_data['CATEGORY']
X_test, y_test = test_data['TITLE'], test_data['CATEGORY']

# Define pipeline components
vectorizer = CountVectorizer(stop_words='english', min_df=5)  # Adjust parameters
classifiers = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Support Vector Machine', SVC()),
    ('Random Forest', RandomForestClassifier())
]

# Define hyperparameters grid for each classifier
param_grids = {
    'Logistic Regression': {'classifier__C': [0.001, 0.01, 0.1, 1, 10]},
    'Multinomial Naive Bayes': {'classifier__alpha': [0.1, 0.5, 1.0]},
    'Support Vector Machine': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'Random Forest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]}
}

# Grid search with cross-validation to find the best model
best_model = None
best_score = 0.0

for name, classifier in classifiers:
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    if grid_search.best_score_ > best_score:
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        best_classifier_name = name

# Evaluate the best model on the test data
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

print(f'Best classifier: {best_classifier_name}')
print(f'Best parameters: {best_params}')
print(f'Validation Accuracy: {best_score:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
