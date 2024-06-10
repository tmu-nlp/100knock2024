import knock41

for sentence in knock41.sentences: #各文
    for chunk in sentence.chunks: #各チャンク
        for morph1 in chunk.morphs: #各形態素
            if morph1.pos == '動詞': #基本形が動詞かを判定
                part = [] #格（助詞）を入れるリスト
                for src in chunk.srcs: #係り元のチャンクの数字について
                    for morph2 in sentence.chunks[src].morphs: #1つの係り元チャンクの形態素について
                        if morph2.pos == '助詞': #助詞かを判定
                            part.append(morph2.surface) 
                if len(part) > 0: #格があるなら
                    sort_part = sorted(list(set(part))) #辞書順（五十音順）にする。
                    part_line = ' '.join(sort_part) #助詞の間はスペース
                    line = morph1.base + "\t" + part_line #述語と格の間はタブ
                    print(line)
                break  # このbreakで最左のみを処理している