# 53. Prediction

def preprocess_single_headline(headline):
    # Tokenize the headline
    tokens = word_tokenize(headline)
    num_words = len(tokens)
    num_capitals = sum(1 for char in headline if char.isupper())
    num_punctuations = sum(1 for char in headline if char in string.punctuation)

    # Create a DataFrame with the same structure as the training features
    feature_dict = {
        'NUM_WORDS': [num_words],
        'NUM_CAPITALS': [num_capitals],
        'NUM_PUNCTUATIONS': [num_punctuations]
    }
    return pd.DataFrame(feature_dict)


def predict_headline_category(headline):
    # Preprocess the headline
    features_df = preprocess_single_headline(headline)

    # Predict the category
    predicted_category_index = model.predict(features_df)[0]
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]

    # Compute prediction probabilities
    prediction_probabilities = model.predict_proba(features_df)[0]

    return predicted_category, prediction_probabilities


# Example usage
headline = "Europe reaches crunch point on banking union"
predicted_category, prediction_probabilities = predict_headline_category(headline)
print(f"Predicted Category: {predicted_category}")
print(f"Prediction Probabilities: {prediction_probabilities}")

# Print probabilities for each category
for category, probability in zip(label_encoder.classes_, prediction_probabilities):
    print(f"{category}: {probability:.4f}")
