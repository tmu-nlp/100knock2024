import random

def shuffle_list(string): 
  n = len(string)
  ls = list(string)
  for i in range(n - 1):
    j = random.randrange(i, n)
    ls[i], ls[j] = ls[j], ls[i]
  return "".join(ls)

def make_typoglycemia(sentence): 
  words = sentence.split(' ')
  for i in range(len(words)):
    word = words[i]

    if (len(word) > 4):
      w = word[0]+ shuffle_list(word[1:-1]) + word[-1] 
      words[i] = w
      
  return " ".join(words)

print(make_typoglycemia("I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."))
  