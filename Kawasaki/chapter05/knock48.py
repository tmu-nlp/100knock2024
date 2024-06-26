import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == "名詞":  # chunkが名詞を含む文節である場合
                path = []
                base_noun = ""
                for morph2 in chunk.morphs: #名詞を含むチャンクの記号以外の形態素をつなげる
                    if morph2.pos != "記号":
                        base_noun += morph2.surface
                path.append(base_noun) #pathに入れる
                while chunk.dst != -1: #係り先がなくなるまで
                    noun = ""
                    for morph3 in sentence.chunks[chunk.dst].morphs: #係り先のチャンクの形態素について
                        if morph3.pos != "記号":
                            noun += morph3.surface
                    path.append(noun)
                    chunk = sentence.chunks[chunk.dst] #新しい係り先
                print(" -> ".join(path))
                break