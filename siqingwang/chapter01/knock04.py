# 04.Atomic symbols
sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can"

words = sentence.split()
indexes = [1, 5, 6, 7, 8, 9, 15, 16, 19]

mapping = {}

for i, word in enumerate(words, 1):
    if i in indexes:
        key = word[0]
    else:
        key = word[:2]
    mapping[key] = i

print(mapping)

words = sentence.split()
for item in enumerate(words, 1):
    print(item)
