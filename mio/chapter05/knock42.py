#42. 係り元と係り先の文節の表示
#係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．
# ただし，句読点などの記号は出力しないようにせよ．



#復習：dst：係り先文節インデックス番号，srcs係り元文節インデックス番号のリスト
#方針１：文が一つずつ格納されたリストsentencesから1つずつ文を取り出す
#方針２：文中の文節が係り先をもつなら、係り元と係り先の文節のテキストを抽出
#方針３：形態素のリストから記号を除外し、文節のテキストを連結
#方針４：係り元と係り先の文節のテキストをタブ区切りで出力

import knock41 

#for sentence in knock41.sentences:
#sentence は Sentenceクラスのインスタンスであるsentence(一文全体)
#sentence.chunks：Chunkクラスのインスタンスのリスト/その文に含まれるすべての文節をもつ


for sentence in knock41.sentences:
   
    #sentence.chunksリストに含まれる各Chunkクラスのインスタンスを取り出す
    for chunk in sentence.chunks: 
        #文中の文節が係り先をもつなら
        if chunk.dst != -1: 
            modifier= "" #係り元                                         #modifier = [] 
            modifee = "" #係り先                                         #modifee = [] 
            
            #係り元の文節の形態素のうち、記号以外の表層形を連結
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modifier +=morph.surface                            #modifier.append(morph.surface)     
                    
            
            #係り先の文節の形態素のうち、記号以外の表層形を連結
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                   modifee +=morph.surface                              #modifee.append(morph.surface)                  
            
            #係り元の文節と係り先の文節をタブ区切りで出力                   #形態素に対する全ての処理の後に一気に文字列を作成
            print(modifier+"\t"+modifee+"\n")                           #print("".join(modifier)+"\t"+"".join(modifee))             
            