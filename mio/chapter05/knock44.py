import knock41
from graphviz import Digraph 


count = 0

for sentence in  knock41.sentences:
    dg = Digraph(format='png')
    for chunk in sentence.chunks:
        if chunk.dst != -1:
            modifier= []
            modifee = []
            
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    #modifier +=morph.surface
                    modifier.append(morph.surface)     
            
            for morph in sentence.chunks[chunk.dst].morphs:
                if morph.pos != "記号":
                    #modifee +=morph.surface
                    modifee.append(morph.surface)     
            phrasein = ''.join(modifier)
            phraseout = ''.join(modifee)
            dg.edge(phrasein, phraseout)
            #dg.edge(modifier, modifee)
            # print(f"{phrasein}\t{phraseout}")
    #output_path = "/home/mohasi/HelloWorld/100knock2024/mio/chapter05/output44.png" 
    #dg.render(output_path+ str(count))
    
    output_path = f"output44_{count}.png"
    dg.render(output_path+ str(count))
    if count <4:
        count += 1
        
    
    
