# 02. “shoe” + “cold” = “schooled”
word1 = "shoe"
word2 = "cold"

concatenated_string = ""
for char1, char2 in zip(word1, word2):
    concatenated_string += char1 + char2

print(concatenated_string)
