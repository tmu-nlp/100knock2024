"""
かかり先:文節解析において、ある文節が他の文節にかかる先を指す。「私は本を読んでいます」であれば、私はが本を読んでいますのかかり先になる。
かかり元:文節解析においてある文節が他の文節からかかられている元を差し、先の例ならば本を読んでいますが私はのかかり元である
"""

class Chunk(object):
    #文節を表すオブジェクトを定義
    def __init__(self,bun,num,chunk_list):
        #文節の形態素リスト
        self.morphs = bun
        #かかり先の文節のインデックス
        self.dst = num
        #かかり元の文節のリスト
        self.srcs = chunk_list
with open ('100knock2024/naoki/chapter05/ai.ja.txt.parsed','r') as f:
    lines = f.readlines()
    all_sentense = []
    sentense = []
    Flag = 0
    chunk_dic = {}
    pnum = -2
    bnum = -2 #初期値
    for text in lines:
        #行の先頭に文字がある場合
        if text[0] == '*':
            #pnumというキーがchunk_dicに存在していない場合、そのキーを追加し、値として空のリスト[]を設定する。bnumをpnumのキーに対応するリストに追加
            chunk_dic.setdefault(pnum,[]).append(bnum)
            #chunk_dicにbnumがないとき空のリストを追加
            if bnum not in chunk_dic:
                chunk_dic[bnum] = []
            #sentenseが空でない場合、それをall_sentenseに追加する
            if sentense:
                all_sentense.appned(Chunk(sentense,pnum,chunk_dic[bnum]).__dict__)
            #かかり先辞書をリセット
            sentense = []
            pnum = int(text.split(" ")[2][:-1])
            bnum = int(text.split(' ')[1])
            if Flag == 1:
                chunk_dic = {}
                Flag = 0
            continue
        if text[0:3] == 'EOS':
            Flag = 1
        else:
            word = text.split('\t')
            sentense.appned(word[0])
            