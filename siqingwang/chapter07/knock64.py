# knock64


# Download the analogy dataset from Gensim
# !wget https://raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt -O questions-words.txt

# Load the dataset into a pandas DataFrame
analogies = []
with open('questions-words.txt', 'r') as file:
    for line in file:
        if line.startswith(":"):
            category = line.strip()
        else:
            words = line.strip().split()
            if len(words) == 4:
                analogies.append((category, words[0], words[1], words[2], words[3]))

analogies_df = pd.DataFrame(analogies, columns=['Category', 'Word1', 'Word2', 'Word3', 'Word4'])


# Function to perform vector arithmetic and find most similar word
def find_analogy_result(word1, word2, word3):
    try:
        result_vector = model[word2] - model[word1] + model[word3]
        most_similar_word, similarity = model.similar_by_vector(result_vector, topn=1)[0]
        return most_similar_word, similarity
    except KeyError as e:
        return None, None

# Apply the function to each row of the DataFrame
analogies_df['Predicted_Word'] = analogies_df.apply(
    lambda row: find_analogy_result(row['Word1'], row['Word2'], row['Word3'])[0], axis=1)
analogies_df['Similarity'] = analogies_df.apply(
    lambda row: find_analogy_result(row['Word1'], row['Word2'], row['Word3'])[1], axis=1)

# Save the results to a new file
analogies_df.to_csv('questions-words-with-predictions.csv', index=False)