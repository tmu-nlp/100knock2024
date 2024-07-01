# knock66

import pandas as pd
from scipy.stats import spearmanr

# Download the WordSimilarity-353 dataset
# !wget http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip -O wordsim353.zip
# !unzip wordsim353.zip

# Load the dataset into a pandas DataFrame
wordsim_df = pd.read_csv('wordsim353/combined.csv')

# Function to compute similarity score from word vectors
def compute_similarity(word1, word2):
    try:
        return model.similarity(word1, word2)
    except KeyError:
        return None

# Compute the similarity scores from word vectors
wordsim_df['Vector_Similarity'] = wordsim_df.apply(
    lambda row: compute_similarity(row['Word 1'], row['Word 2']), axis=1)

# Drop rows where similarity could not be computed
wordsim_df = wordsim_df.dropna(subset=['Vector_Similarity'])

# Compute the Spearman's rank correlation coefficient
human_similarity_scores = wordsim_df['Human (mean)']
vector_similarity_scores = wordsim_df['Vector_Similarity']
spearman_correlation, _ = spearmanr(human_similarity_scores, vector_similarity_scores)

print(f"Spearman's rank correlation coefficient: {spearman_correlation}")
