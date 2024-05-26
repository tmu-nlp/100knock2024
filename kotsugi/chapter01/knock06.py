def n_gram(word, n):
  return { word[i:i+n] for i in range(len(word)) }

x = n_gram('paraparaparadise', 2)
y = n_gram('paragraph', 2)


print(f'x | y = {x | y}')

print(f'x & y = {x & y}')
print(f'x - y = {x - y}')
print(f'"se" in x = {"se" in x}')
print(f'"se" in y = {"se" in y}')
