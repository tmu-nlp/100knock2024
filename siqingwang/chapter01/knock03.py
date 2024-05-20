# 03. Pi
sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics"
word_list = sentence.split()

print(word_list)

# Count the number of alphabetical letters in each word
letter_counts = [sum(c.isalpha() for c in word) for word in word_list]

print(letter_counts)
