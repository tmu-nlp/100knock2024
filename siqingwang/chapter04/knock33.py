words = morphology_map(file_parsed)
for word in words:
    if word['pos'] == '名詞' and word['pos1'] == 'サ変接続':
        items.add(word['surface'])
        items_order.append(word['surface'])