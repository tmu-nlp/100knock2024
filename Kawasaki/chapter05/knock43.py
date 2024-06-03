import knock41

#knock42.pyと基本同じ。

for sentence in knock41.sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            normbool = 0 #名詞を含む文節かの判定
            verbbool = 0 #動詞を含む文節かの判定
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modiin.append(morph.surface)
                if morph.pos == "名詞":
                    normbool = 1
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    modifor.append(morph.surface)
                if morph.pos == "動詞":
                    verbbool = 1
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            if normbool and verbbool:
                print(f"{phrasein}\t{phraseout}")