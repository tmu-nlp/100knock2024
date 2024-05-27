suf_list = []
for sentense in morphemes:
    for text in sentense:
        if text["pos"] == "動詞":
            suf_list.append(text["surface"])
verb = set(suf_list)
verb 