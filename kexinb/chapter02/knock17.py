# task 17. １列目の文字列の異なり
# 1列目の文字列の種類（異なる文字列の集合）を求めよ．確認にはcut, sort, uniqコマンドを用いよ．

import pandas as pd

def unique_names(path):
    df = pd.read_table(path, header=None)
    uniq = df[0].unique()
    uniq.sort() # in-place
    return uniq

if __name__ == "__main__":
    out = unique_names("data/popular-names.txt")
    print(out)