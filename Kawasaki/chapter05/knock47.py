import knock41

for sentence in knock41.sentences:  
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':  # 最左動詞を発見
                for i, src1 in enumerate(chunk.srcs):  # 動詞にかかっている文節を調査
                    if len(sentence.chunks[src1].morphs) == 2 and sentence.chunks[src1].morphs[0].pos1 == "サ変接続" and sentence.chunks[src1].morphs[1].surface == "を":
                        #条件式１：係り元のチャンクの要素数が2かを判定
                        #条件式２：係り元のチャンクの1番目の形態素がサ変接続かを判定
                        #条件式３：係り元のチャンクの2番目の形態素が「を」か判定
                        trg_pred = sentence.chunks[src1].morphs[0].surface + \
                            sentence.chunks[src1].morphs[1].surface + \
                            morph1.base
                        # trg_predは述語（サ変接続名詞＋を＋動詞の基本形）
                        part = []
                        frame = ""
                        num = 0
                        for src2 in chunk.srcs[:i] + chunk.srcs[i+1:]:  # i番目のチャンク（サ変接続名詞＋を）は抜かす
                            for morph2 in sentence.chunks[src2].morphs:
                                if morph2.pos == "助詞":
                                    frame = ""
                                    for morph3 in sentence.chunks[src2].morphs:
                                        if morph3.pos != '記号':
                                            frame += morph3.surface
                                    part.append((morph2.surface, frame))
                        if len(part) > 0:
                            sort_part = sorted(
                                list(set(part)), key=lambda x: x[0])
                            pattern_case = ""
                            frame_case = ""
                            for row in sort_part:
                                pattern_case += row[0] + " "
                                frame_case += row[1] + " "
                            frame_case = frame_case.rstrip(" ")
                            line = trg_pred + "\t" + pattern_case + "\t" + frame_case
                            print(line)
                break