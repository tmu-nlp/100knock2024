# task 13. col1.txtとcol2.txtをマージPermalink
# 12で作ったcol1.txtとcol2.txtを結合し，元のファイルの1列目と2列目をタブ区切りで並べたテキストファイルを作成せよ．確認にはpasteコマンドを用いよ

import pandas as pd
import filecmp

def merge_cols(path1, path2):
    df1, df2 = pd.read_table(path1, header=None),\
               pd.read_table(path2, header=None)
    return pd.concat([df1, df2], axis=1)

if __name__ == "__main__":
    df = merge_cols("data/popular-names-col1.txt", "data/popular-names-col2.txt")
    df.to_csv("data/popular-names-merged.txt", sep='\t', index=False, header=False)

    assert filecmp.cmp("data/popular-names-merged.txt", "data/popular-names-merged-unix.txt")
