# task 31. 動詞
# 動詞の表層形をすべて抽出せよ．

from typing import Dict, List
from knock30 import parse_mecab

def extract_attr(sentences: List[List[Dict]], pos: str, attr:str) -> List[str]:
    result = []
    for sentence in sentences:
        for morph in sentence:
            if morph['pos'] == pos:
                result.append(morph[attr])
    return result

if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        print(*extract_attr(output, "動詞", "surface"), sep="\n")