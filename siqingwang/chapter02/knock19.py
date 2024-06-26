# 19. Frequency of a string in the first column in descending orderPermalink

frequency = df[0].value_counts()
sorted_frequency = frequency.sort_values(ascending=False)

print(sorted_frequency)