from gensim.models import KeyedVectors
import pickle
#jpblibとは
#CBoW & nagativesampling

model = KeyedVectors.load_word2vec_format("drive/MyDrive/GoogleNews-vectors-negative300.bin.gz", binary=True)

with open("drive/MyDrive/word2vec.pkl", "wb") as f:
    pickle.dump(model, f)

print(model["United_States"])