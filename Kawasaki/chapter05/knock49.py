import knock41
from itertools import combinations
import re

for sentence in knock41.sentences:
    nouns = []

    #一つの文のチャンクに名詞があるかを判定、あればそのチャンクの番号をnounsにappend
    for i, chunk in enumerate(sentence.chunks):
        for morph1 in chunk.morphs:
            if morph1.pos == "名詞":
                nouns.append(i)
                break

    for i, j in combinations(nouns, 2):  # nounsから任意の2つの組み合わせを獲得、i,jは名詞節番号が格納
        
        #係り先はどこかで一致する！
        
        path_i = [] #iの係り先を記録
        path_j = [] #jの係り先を記録

        #小さい番号からpath_{hoge}を計算して次の係り先に移動
        #i == j となるまで(ペアの番号iとjの係り先をたどって一致するまで)続ける
        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence.chunks[i].dst
            else:
                path_j.append(j)
                j = sentence.chunks[j].dst

        #パターン１：文節iから構文木の根に至る経路上に文節jが存在する場合
        if len(path_j) == 0:  
             
            #最初の名詞節path_i[0]
            chunkX1 = ""
            #名詞をXに置換する
            for morph2 in sentence.chunks[path_i[0]].morphs:
                if morph2.pos != "名詞" and morph2.pos != "記号": #名詞でないならそのまま追加
                    chunkX1 += morph2.surface
                elif morph2.pos == "名詞": #名詞ならXに変える
                    chunkX1 += 'X'
            
            # 最後の名詞節i
            chunkY1 = ""
            # 名詞をYに置換する
            for morph3 in sentence.chunks[i].morphs:
                if morph3.pos != "名詞" and morph3.pos != "記号": #名詞でないならそのまま追加
                    chunkY1 += morph3.surface
                elif morph3.pos == "名詞": #名詞ならYに変える
                    chunkY1 += 'Y'

            chunkX1_sub = re.sub('X+', 'X', chunkX1) #X+をxに変える
            chunkY1_sub = re.sub('Y+', 'Y', chunkY1) #Y+をYに変える

            #path_i[0]からiまでの経路上の文節を用意
            pathway = []
            for n in path_i[1:]:
                route_chunk = ""
                for morph4 in sentence.chunks[n].morphs:
                    if morph4.pos != "記号":
                        route_chunk += morph4.surface
                pathway.append(route_chunk)
            pathX2Y = [chunkX1_sub] + pathway + [chunkY1_sub] #全てを結合
            print(' -> '.join(pathX2Y))

        #パターン２：文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合
        else:  
            # 文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示
            
            #最初の名詞節path_i[0]
            chunkX2 = ""
            for morph5 in sentence.chunks[path_i[0]].morphs:
                if morph5.pos != "名詞" and morph5.pos != "記号":
                    chunkX2 += morph5.surface
                elif morph5.pos == "名詞":
                    chunkX2 += 'X'

            #最初の名詞節path_j[0]
            chunkY2 = ""
            for morph6 in sentence.chunks[path_j[0]].morphs:
                if morph6.pos != "名詞" and morph6.pos != "記号":
                    chunkY2 += morph6.surface
                elif morph6.pos == "名詞":
                    chunkY2 += 'Y'

            # 合流地点の文節i
            chunk_k = "" 
            for morph7 in sentence.chunks[i].morphs:
                if morph7.pos != "記号":
                    chunk_k += morph7.surface
            
            chunkX2_sub = re.sub('X+', 'X', chunkX2) #X+をXに置き換え
            chunkY2_sub = re.sub('Y+', 'Y', chunkY2) #Y+をYに置き換え
            
            #path_i[0]からiまでの経路上の文節を用意
            #path_j[0]からiまでの経路上の文節を用意
            
            pathwayX = []
            pathwayY = []

            for n in path_i[1:]:
                route_chunk_x = ""
                for morph8 in sentence.chunks[n].morphs:
                    if morph8.pos != "記号":
                        route_chunk_x += morph8.surface
                pathwayX.append(route_chunk_x)
            for n in path_j[1:]:
                route_chunk_y = ""
                for morph9 in sentence.chunks[n].morphs:
                    if morph9.pos != "記号":
                        route_chunk_y += morph9.surface
                pathwayY.append(route_chunk_y)
            path_X = [chunkX2_sub] + pathwayX
            path_Y = [chunkY2_sub] + pathwayY

            print(' | '.join(
                [' -> '.join(path_X), ' -> '.join(path_Y), chunk_k]))

            # print(f"path_X : {path_X}")
            # print(f"path_Y : {path_Y}")
            # print(f"chunk_k 