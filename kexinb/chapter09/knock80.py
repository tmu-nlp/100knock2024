# task80. ID番号への変換
#問題51で構築した学習データ中の単語にユニークなID番号を付与したい.
#学習データ中で最も頻出する単語に1,2番目に頻出する単語に2,……といった方法で, 学習データ中で2回以上出現する単語にID番号を付与せよ.
#そして,与えられた単語列に対して,ID番号の列を返す関数を実装せよ.ただし,出現頻度が2回未満の単語のID番号はすべて0とせよ.

import pandas as pd
from collections import defaultdict
import string

class ID():
    def __init__(self, data):
        self.train_dict = defaultdict(int)  # Dictionary to count word frequencies
        self.id_dict = {}  # Dictionary to map words to IDs
        self.make_id(data)  # Initialize the ID assignment process

    def make_id(self, data):
        # Combine all lines into a single string, replace punctuation, then split into words
        all_words = ' '.join(data).translate(table).split()

        # Count word frequencies
        for word in all_words:
            self.train_dict[word] += 1

        # Sort words by frequency
        sort_list = sorted(self.train_dict.items(), key=lambda x: x[1], reverse=True)

        # Assign IDs based on frequency
        for i, (word, freq) in enumerate(sort_list):
            self.id_dict[word] = i + 1 if freq >= 2 else 0

    def return_id(self, line):
        # Translate punctuation and split line into words
        words = line.translate(table).split()
        # Map words to IDs, default to 0 if word not found
        return [self.id_dict.get(word, 0) for word in words]

# Translation table to replace punctuation with spaces
table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

# File and column setup
header_name = ['TITLE', 'CATEGORY']
train_file = "output/ch6/train.txt"
train_data = pd.read_csv(train_file, header=None, sep='\t', 
                         names=header_name)['TITLE']

word_ids = ID(train_data)  # Initialize ID class with training titles
test_vec = word_ids.return_id(train_data[10])  # Get IDs for the 10th title

if __name__ == "__main__":
    # Print the word-ID dictionary
    top_words = list(word_ids.id_dict.items())[:20]
    for word, id_ in top_words:
        print(f"{word} : {id_}")

'''
to : 1
s : 2
in : 3
UPDATE : 4
on : 5
as : 6
US : 7
of : 8
for : 9
The : 10
1 : 11
To : 12
2 : 13
the : 14
and : 15
In : 16
Of : 17
at : 18
a : 19
A : 20
'''