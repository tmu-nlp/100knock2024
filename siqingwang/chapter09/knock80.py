import pandas as pd

data = pd.read_csv('newsCorpora.csv', sep = '\t', header = None, names = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

publishers = ['Reuters', 'Huffington Post', 'Businessweek', '“Contactmusic.com', 'Daily Mail']

data = data[data['PUBLISHER'].isin(publishers)]
data = data[['TITLE', 'CATEGORY']]

from collections import Counter
import pandas as pd


class WordToIDMapper:
    def __init__(self):
        self.word_to_id = {}
        self.id_counter = 1  # Start IDs from 1 (0 will be reserved for words occurring less than twice)

    def fit_from_dataframe(self, df, column_name):
        # Extract words from the specified column in the DataFrame
        # 特定されるcolumn(TITLE)から単語をsplit
        data = df[column_name].str.split().sum()  # Split titles into words and flatten into a list

        # Count word frequencies
        # 頻度を計算
        word_counts = Counter(data)

        # Sort words by frequency
        # 頻度から順番を付け
        # based on the second element of each tuple
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Assign IDs to words that occur two or more times
        for word, count in sorted_words:
            if count >= 2:
                self.word_to_id[word] = self.id_counter
                self.id_counter += 1

    def get_id(self, word):
        # Return the ID of a word if it exists, otherwise return 0
        return self.word_to_id.get(word, 0)


# Initialize the mapper
mapper = WordToIDMapper()

# Fit the mapper to your DataFrame
mapper.fit_from_dataframe(data, 'TITLE')


# Function to get ID for a word
def get_word_id(word):
    return mapper.get_id(word)

text = data.iloc[1, data.columns.get_loc('TITLE')]
words = text.split()
word_ids = [get_word_id(word) for word in words]

print(text)
print(word_ids)