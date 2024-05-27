# 06.Set
def extract_letter_bigrams(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

# Extract letter bi-grams from the words
word1 = "paraparaparadise"
word2 = "paragraph"

X = set(extract_letter_bigrams(word1))
Y = set(extract_letter_bigrams(word2))

# Union
union = X.union(Y)

# Intersection
intersection = X.intersection(Y)

# Difference
difference = X.difference(Y)

# Check if "se" is included in sets X and Y
se_in_X = "se" in X
se_in_Y = "se" in Y

print("Union:", union)
print("Intersection:", intersection)
print("Difference:", difference)
print("Is 'se' in X?", se_in_X)
print("Is 'se' in Y?", se_in_Y)