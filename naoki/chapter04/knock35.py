import collections
word_list = []
for sentense in morphemes:
    for text in sentense:
        if text['pos'] != '補助記号':
            word_list.append(text['surface'])
word_list_count = collections.Counter(word_list)
word_list_rank = word_list_count.most_common()
word_list_rank