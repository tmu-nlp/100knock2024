# knock65

# Compute accuracy for semantic and syntactic analogies
semantic_analogies = analogies_df[analogies_df['Category'].str.contains('semantic')]
syntactic_analogies = analogies_df[analogies_df['Category'].str.contains('gram')]

# Calculate accuracy
semantic_accuracy = (semantic_analogies['Word4'] == semantic_analogies['Predicted_Word']).mean()
syntactic_accuracy = (syntactic_analogies['Word4'] == syntactic_analogies['Predicted_Word']).mean()

print(f"Semantic analogy accuracy: {semantic_accuracy}")
print(f"Syntactic analogy accuracy: {syntactic_accuracy}")