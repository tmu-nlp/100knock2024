# task 32. 動詞の基本形
# 動詞の基本形をすべて抽出せよ．

from typing import Dict, List
from knock30 import parse_mecab
from knock31 import extract_attr

if __name__ == "__main__":
    with open('neko.txt.mecab','r') as f:
        nekoData = f.read()
        output = parse_mecab(nekoData)
        print(*extract_attr(output, "動詞", "base"), sep="\n")