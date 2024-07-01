# knock61

from numpy import dot
from numpy.linalg import norm

word_vector = model['United_States']

vector_united_states = model['United_States']
vector_us = model['U.S.']

# Compute cosine similarity
cosine_similarity = dot(vector_united_states, vector_us) / (norm(vector_united_states) * norm(vector_us))

# Print the cosine similarity
print(f"Cosine similarity between 'United States' and 'U.S.': {cosine_similarity}")