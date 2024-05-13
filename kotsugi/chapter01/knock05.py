def n_gram(word, n):
  gram = []
  for i in range(len(word)):
    gram.append(word[i:i+n])
  return gram

print(n_gram("I am an NLPer".split(" "), 2))
print(n_gram("I am an NLPer", 2))
