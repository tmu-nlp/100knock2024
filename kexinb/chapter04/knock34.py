# task34. 名詞の連接
# 名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

from knock30 import parse_mecab

def extract_consecutive(sentences, pos):
    result = []
    for sentence in sentences:
        currPhrase, cnt = "", 0
        for morph in sentence:
            if morph["pos"] == pos:
                cnt += 1
                currPhrase += morph["surface"]
            else:
                if cnt > 1:
                    result.append(currPhrase)
                currPhrase, cnt = "", 0
        # Check EOS for remaining noun phrase
        if cnt > 1:
            result.append(currPhrase)
    return result

if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        print(*extract_consecutive(output,"名詞"), sep="\n")




# def longestConsecutive(sentences, pos):
#     bestLen = 0
#     bestPhrases = []

#     for sentence in sentences:
#         currPhrase = ""
#         for morph in sentence:
#             if morph['pos'] == pos:
#                 currPhrase += morph['surface']
#             else:
#                 if len(currPhrase) > bestLen:
#                     bestLen = len(currPhrase)
#                     bestPhrases = [currPhrase]  # Start a new list with currPhrase
#                 elif len(currPhrase) == bestLen:
#                     bestPhrases.append(currPhrase)
#                 currPhrase = ""  # Reset currPhrase after encountering non-noun

#         # Check the last phrase in the sentence
#         if len(currPhrase) > bestLen:
#             bestLen = len(currPhrase)
#             bestPhrases = [currPhrase]
#         elif len(currPhrase) == bestLen:
#             bestPhrases.append(currPhrase)

#     return bestPhrases