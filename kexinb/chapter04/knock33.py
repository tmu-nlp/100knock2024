# task33. 「AのB」
# 2つの名詞が「の」で連結されている名詞句を抽出せよ．

from knock30 import parse_mecab

def extract_noun_no_noun(sentences):
    result = []
    for sentence in sentences:
        for i in range(1, len(sentence)-1):
            if sentence[i]['surface'] == "の" \
                and sentence[i-1]["pos"] == "名詞" \
                    and sentence[i+1]["pos"] == "名詞":
                result.append(sentence[i-1]["surface"] + 
                              sentence[i]["surface"] +
                              sentence[i+1]["surface"])
    return result

if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        print(*extract_noun_no_noun(output), sep="\n")



# def extract_noun_no_noun(sentences):
#     result = [
#         f"{sentence[i-1]['surface']}の{sentence[i+1]['surface']}"
#         for sentence in sentences
#         for i in range(1, len(sentence) - 1)
#         if sentence[i]['surface'] == "の" and
#            sentence[i-1]['pos'] == "名詞" and
#            sentence[i+1]['pos'] == "名詞"
#     ]   # faster maybe?
#     return result