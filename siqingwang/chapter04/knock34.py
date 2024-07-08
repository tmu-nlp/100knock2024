items_list = []
words = morphology_map(file_parsed)
for i in range(1, len(words) - 1):
    if words[i]['surface'] == 'の' and words[i - 1]['pos'] == '名詞' and words[i + 1]['pos'] == '名詞':
            items_list.append(words[i - 1]['surface'] + 'の' + words[i + 1]['surface'])

items = set(items_list)
items = sorted(items, key=items_list.index)
for item in items:
    print(item)