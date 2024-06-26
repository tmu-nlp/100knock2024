# 51. Feature extraction

import knock50
from knock50 import train
from knock50 import valid
from knock50 import  test

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string

# Download NLTK data files (you might need to run this once)
nltk.download('punkt')

def extract_features(data):
    features = []
    for _, row in data.iterrows():
        headline = row['TITLE']
        tokens = word_tokenize(headline)
        num_words = len(tokens)
        num_capitals = sum(1 for char in headline if char.isupper())
        num_punctuations = sum(1 for char in headline if char in string.punctuation)
        features.append({
            'CATEGORY': row['CATEGORY'],
            'TOKENS': ' '.join(tokens),
            'NUM_WORDS': num_words,
            'NUM_CAPITALS': num_capitals,
            'NUM_PUNCTUATIONS': num_punctuations
        })
    return pd.DataFrame(features)



# Extract features
train_features = extract_features(train)
valid_features = extract_features(valid)
test_features = extract_features(test)


# Print a sample of the features to verify
print("Training features sample:")
print(train_features.head())

print("\nValidation features sample:")
print(valid_features.head())

print("\nTest features sample:")
print(test_features.head())

train_features.to_csv('train.feature.txt', sep='\t', index=False)
valid_features.to_csv('valid.feature.txt', sep='\t', index=False)
test_features.to_csv('test.feature.txt', sep='\t', index=False)