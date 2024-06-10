import knock41

#knock45.pyと違うとこだけ

for sentence in knock41.sentences:  
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':
                part = [] #格と項のタプルで入れるリスト
                for src in chunk.srcs: 
                    for morph2 in sentence.chunks[src].morphs:
                        if morph2.pos == '助詞':
                            frame = ""
                            for morph3 in sentence.chunks[src].morphs:
                                if morph3.pos != '記号':
                                    frame += morph3.surface
                            part.append((morph2.surface, frame)) #morp
                if len(part) > 0:
                    sort_part = sorted(list(set(part)), key=lambda x: x[0])
                    pattern_case = ""
                    frame_case = ""
                    for row in sort_part:
                        pattern_case += row[0] + " "
                        frame_case += row[1] + " "
                    frame_case = frame_case.rstrip(" ")
                    line = morph1.base + "\t" + pattern_case + frame_case
                    print(line)
                break  # このbreakで最左のみを処理している