with open("C:/Users/shish_sf301y1/Desktop/pyファイル/neko.txt.mecab", "r") as f:
    morphemes = []
    neko_list = []
    lines = f.readlines()
    for line in lines:
        neko_dic = {}
        suf = line.split("\t")
        if suf[0] == "EOS\n": 
            continue
        #suf[1]には名詞,普通名詞,副詞可能,,,,トキドキ,時々,時々,...
        temp = suf[1].split(',')
        neko_dic["surface"] = suf[0]
        #なぜ7かは不明
        if len(temp) <= 7:
            neko_dic["base"] = suf[0]
        else :
            neko_dic["base"] = temp[7]
        neko_dic["pos"] = temp[0]
        neko_dic["pos1"] = temp[1]
        neko_list.append(neko_dic)
        if suf[0] == "。":
            morphemes.append(neko_list)
            neko_list = []
morphemes   