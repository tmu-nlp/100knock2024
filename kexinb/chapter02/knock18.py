# task18. 各行を3コラム目の数値の降順にソート
# 各行を3コラム目の数値の逆順で整列せよ（注意: 各行の内容は変更せずに並び替えよ）．確認にはsortコマンドを用いよ（この問題はコマンドで実行した時の結果と合わなくてもよい）.

import pandas as pd

def sort_by_col(col:int, asc:bool, path:str):
    df = pd.read_table(path, header=None)
    return df.sort_values(col, ascending=asc)

if __name__ == "__main__":
    out = sort_by_col(2, False, "data/popular-names.txt")
    out.to_csv("data/popular-names-sortbycol3-rev.text", index=None, header=None, sep='\t')
