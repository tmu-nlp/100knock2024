# task19. 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
# 各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．

import pandas as pd

def sort_by_freq(col,path):
    df = pd.read_table(path, header=None)
    return df[col].value_counts()

if __name__ == "__main__":
    out = sort_by_freq(0, "data/popular-names.txt")
    out.to_csv("data/popular-names-namesbyfreq.text", index=None, header=None, sep='\t')
    print(out[0:10])