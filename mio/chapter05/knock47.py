#機能動詞構文のマイニングPermalink
#動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．46のプログラムを以下の仕様を満たすように改変せよ．

#「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
#述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
#述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
#述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）
#例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，以下の出力が得られるはずである．
#学習を行う	に を	元に 経験を

import knock41

for sentence in knock41.sentences:  # 一文を選択
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == '動詞':  # 最左動詞があったら
                for i, src1 in enumerate(chunk.srcs):  # 動詞にかかっている文節を調査
                        #文節の長さが2か？ 最初の形態素がサ変接続か？ #2番目の形態素が「を」か？
                    if len(sentence.chunks[src1].morphs) == 2 and \
                            sentence.chunks[src1].morphs[0].pos1 =="サ変接続" and \
                            sentence.chunks[src1].morphs[1].surface == "を":
                        trg_pred = sentence.chunks[src1].morphs[0].surface + \
                                   sentence.chunks[src1].morphs[1].surface + \
                                   morph1.base
                        # 述語の準備が完了
                        part = [] #助詞とその文節全体を格納
                        #動詞にかかっている文節のうち、i番目の文節を除外してループ 
                        for src2 in chunk.srcs[:i] + chunk.srcs[i+1:]:  #chunk.srcs：動詞にかかる文節のインデックスのリスト
                            for morph2 in sentence.chunks[src2].morphs: #文節内の形態素をループで回す
                                if morph2.pos == "助詞": 
                                    frame = "" #文節全体を保持する文字列を初期化
                                    for morph3 in sentence.chunks[src2].morphs: #文節内の形態素を再びループで回す
                                        if morph3.pos != '記号':
                                            frame += morph3.surface
                                            #(助詞, 文節) のタプルを追加
                                    part.append((morph2.surface, frame)) #助詞と助詞を含む文節をタプルとして part リストに追加
                        if len(part) > 0:
                            # 助詞でソートし、重複を排除
                            sort_part = sorted(set(part), key=lambda x: x[0])
                            pattern_case = " ".join(p[0] for p in sort_part)
                            frame_case = " ".join(p[1] for p in sort_part)
                            line = f"{trg_pred}\t{pattern_case}\t{frame_case}"
                            print(line)
                break  # このbreakにより最左のみを処理