#neko.txt.mecabを作成する．
import MeCab
basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter04/"
tagger = MeCab.Tagger()#規定の辞書(unidic)を指定

keitaiso_sentence=[]
with open(basepath+"neko.txt","r")as f_in, open(basepath+"neko.txt.mecab","w")as f_out:
    lines = f_in.readlines()
    for l in lines:
        result = tagger.parse(l)#1文ごとに形態素．
        f_out.write(result)
f_in.close()
f_out.close()