import MeCab
#()のなかには設定オプションをいれられる。
tagger = MeCab.Tagger()
with open("neko.txt") as f:
    #wで新しいファイルを作成できる
    with open("neko.txt.mecab","w")as f2:
        text=f.read().split("\t")
        for texts in text:
            #tagger.parseは文字列にしてから使用する(しないとTypeErrorが起こる)
            #EOS(End Of Sentence)
            cat = tagger.parse(texts).replace("EOS","")
            f2.write(cat)
