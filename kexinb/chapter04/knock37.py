# task37. 「猫」と共起頻度の高い上位10語
# 「猫」とよく共起する（共起頻度が高い）10語とその出現頻度をグラフ（例えば棒グラフなど）で表示せよ．

from knock30 import parse_mecab
import matplotlib.pyplot as plt
import japanize_matplotlib

def cooccur(sentences, word): # occur in the same sentence (fig2)
    cooccurDict = {}

    for sentence in sentences:
        if any(morph["surface"] == word for morph in sentence):
            for morph in sentence:
                currWord = morph["surface"]
                if currWord == word or morph["pos"] == "記号":
                    continue
                if currWord in cooccurDict:
                    cooccurDict[currWord] += 1
                else:
                    cooccurDict[currWord] = 1
    
    result = [(freq, word) for word, freq in cooccurDict.items()]
    result = sorted(result, reverse=True)
    return result

# def cooccur(sentences, word): # directly co-occur (fig1)
#     cooccurDict = {}
#     for sentence in sentences:
#         words = [morph['surface'] for morph in sentence if morph['pos'] != '記号']
#         for i, currWord in enumerate(words):
#             if currWord == word:
#                 if i > 0: # has prev
#                     prevWord = words[i - 1]
#                     cooccurDict[prevWord] = cooccurDict.get(prevWord, 0) + 1
#                 if i < len(words) - 1: # has next
#                     nextWord = words[i + 1]
#                     cooccurDict[nextWord] = cooccurDict.get(nextWord, 0) + 1

#     result = [(freq, word) for word, freq in cooccurDict.items()]
#     result = sorted(result, reverse=True)
#     return result


if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        
        catCooccur = cooccur(output, "猫")
        
        cnts = [i[0] for i in catCooccur]
        chrs = [i[1] for i in catCooccur]
    
    topN = 10
    
    plt.figure(figsize=(10, 8))
    plt.bar(chrs[:topN], cnts[:topN], color='skyblue')
    plt.xlabel('Characters')  
    plt.ylabel('Frequency')  
    plt.title('Frequency of the Top {} Characters Cooccurring with "猫"'.format(topN))
    plt.savefig('knock37_output_2.png')