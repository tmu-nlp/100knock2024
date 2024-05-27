suf_list = []
for sentense in morphemes:
    #最初と最後は取らないように回数を調整する
    for i in range(len(sentense)-2):
        if sentense[i+1]['base'] == 'の' and sentense[i]['pos'] == '名詞' and sentense[i+2]['pos'] == '名詞':
            suf_list.append(sentense[i]['surface']+sentense[i+1]['surface']+sentense[i+2]['surface'])
suf_list