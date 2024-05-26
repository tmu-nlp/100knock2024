#43. 名詞を含む文節が動詞を含む文節に係るものを抽出
#名詞を含む文節が，動詞を含む文節に係るとき，これらをタブ区切り形式で抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．

import knock41

for sentence in knock41.sentences:
    for chunk in sentence.chunks: #sentence.chunksリストに含まれる各Chunkのインスタンスを順に取り出す
        if chunk.dst != -1: #文中の文節が係り先をもつなら
            
            modifier = "" #係り元
            modifiee = "" #係り先
            modifier_pos = ""
            modifee_pos = ""
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modifier= morph.surface
                    modifier_pos= morph.pos   
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    modifiee.append(morph.surface)  
                    modifee_pos.add(morph.pos)   
            
            if "名詞" in modifier_pos and "動詞" in modifee_pos:
                print("".join(modifier)+"\t"+"".join(modifiee) +"\n")             


