# task 12. 1列目をcol1.txtに，2列目をcol2.txtに保存Permalink
# 各行の1列目だけを抜き出したものをcol1.txtに，2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．確認にはcutコマンドを用いよ．

import pandas as pd
import filecmp

def extract_col(idx_col, filepath_in): # extract column at idx_col and write to file
    df = pd.read_table(filepath_in, header=None)
    
    out_col = df.iloc[:, idx_col]
    
    filepath_out = filepath_in.replace('.txt', '-col' + str(idx_col+1) + '.txt')
    out_col.to_csv(filepath_out, index=False, header=None)


if __name__ == '__main__':
    filepath_in = '../data/popular-names.txt'

    extract_col(0, filepath_in)
    extract_col(1, filepath_in)

    # assert filecmp.cmp('../data/popular-names-col1.txt', '../data/popular-names-col1-unix.txt')
    # assert filecmp.cmp('../data/popular-names-col2.txt', '../data/popular-names-col2-unix.txt')
