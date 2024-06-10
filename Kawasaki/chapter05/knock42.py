import knock41

for sentence in knock41.sentences:
    for chunk in sentence.chunks: #1つのチャンクについて
        if chunk.dst != -1: #係り先が存在するか
            modiin = [] #係り元のチャンクの形態素がすべて入るリスト
            modifor = [] #係り先のチャンクの形態素がすべて入るリスト
            for morph in chunk.morphs: #係り元のチャンクについて
                if morph.pos != "記号": #記号でないなら
                    modiin.append(morph.surface) #全て入れていく
            for morph in sentence.chunks[chunk.dst].morphs: #係り先のチャンクについて
                if morph.pos != "記号": #記号でないなら
                    modifor.append(morph.surface) #全て入れていく
            phrasein = ''.join(modiin) #文字列に
            phraseout = ''.join(modifor)
            print(f"{phrasein}\t{phraseout}")