"""
#44.係り受け木の可視化
与えられた文の係り受け木を有向グラフとして可視化せよ．
可視化には，Graphviz等を用いるとよい．
"""

import knock41
import graphviz

pair = []#係り元の文節と係り先の文節のテキストを格納する

#42と同じ：係り元の文節と係り先の文節のテキストを、句読点などの記号は出力しないように、タブ区切り形式で抽出
for chunk in sentences[2].chunks: 
    if int(chunk.dst) == -1: 
        continue            #係り先文節がなければスキップ
    else:
        surf = "".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"]) #係り先文節がある場合、文節内の各表層形を抽出して連結
        next_surf = "".join([morph.surface for morph in sentences[2].chunks[int(chunk.dst)].morphs if morph.pos != "記号"])#係り元の文節を抽出して連結
        pair.append((surf, next_surf)) #抽出した係り元と係り先のペアをpairリストに追加

# Graphvizを使ってグラフを可視化
graph = graphviz.Digraph()
#pairリストに格納された係り元と係り先のペアを使ってエッジを追加
for s, t in pair:
    graph.edge(s, t)

graph.format = 'png'
graph.render('result44')
