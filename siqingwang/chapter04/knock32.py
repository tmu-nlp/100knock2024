words = morphology_map(file_parsed)
for word in words:
    if word['pos'] == '動詞':
        verbs.add(word['base'])
        verbs_order.append(word['base'])