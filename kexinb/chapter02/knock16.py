# task 16. ファイルをN分割する
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．同様の処理をsplitコマンドで実現せよ．

import pandas as pd

def split_n_files(n, path):
    df = pd.read_table(path, header=None)
    print(len(df))
    k, m = divmod(len(df), n)
    return (df[i*k + min(i, m) : (i+1)*k + min(i+1, m)] 
               for i in range(n))

if __name__ == "__main__":
    while True:
        try:
            n = int(input("Number of files to split into: "))
            break
        except ValueError:
            print("Please enter a valid number")

    out = split_n_files(n, "data/popular-names.txt")
    print(list(len(i) for i in out))