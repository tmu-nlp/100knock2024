import knock41

for sentence in knock41.sentences:
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modiin = []
            modifor = []
            for morph in chunk.morphs:
                if morph.pos != '記号':
                    modiin.append(morph.surface)
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != '記号':
                    modifor.append(morph.surface) #ココ工夫したい
            phrasein = ''.join(modiin)
            phraseout = ''.join(modifor)
            print(f'{phrasein}\t{phraseout}')


            
