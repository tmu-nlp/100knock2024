# task35. 単語の出現頻度
# 文章中に出現する単語とその出現頻度を求め，出現頻度の高い順に並べよ．

from knock30 import parse_mecab

def count_words(sentences):
    freqDict = {}
    for sentence in sentences:
        for morph in sentence:
            if morph["pos"] == "記号":
                continue
            word = morph["surface"]
            freqDict[word] = freqDict.get(word, 0) + 1
    
    result = [(freq, word) for word, freq in freqDict.items()]
    result = sorted(result, reverse=True)
    return result



if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        print(*count_words(output), sep="\n")
