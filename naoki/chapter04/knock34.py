suf_list = []
#自然言語処理100本ノックのような名詞を取得
for sentense in morphemes:
    count = 0
    sent = ''
    for i in range(len(sentense)):
        if sentense[i]['pos'] == '名詞' :
            count += 1
            sent += sentense[i]['surface']
        else :
            if count >= 2:
                suf_list.append(sent)
            count = 0
            sent = ''
suf_list = set(suf_list)
suf_list