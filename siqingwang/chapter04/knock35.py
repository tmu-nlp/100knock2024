words = morphology_map(file_parsed)
nouns = []
for word in words:
    if word['pos'] == '名詞':
        nouns.append(word['surface'])
    else:
        if len(nouns) > 1:
            items_list.append("".join(nouns))
        nouns = []