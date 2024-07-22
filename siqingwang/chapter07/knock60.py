# knock60

from gensim.models import KeyedVectors


file = '/content/drive/MyDrive/NLP/GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(file, binary = True)
model['United_States']