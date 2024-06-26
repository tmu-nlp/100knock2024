#  task 14. 先頭からN行を出力
#  自然数Nをコマンドライン引数などの手段で受け取り，入力のうち先頭のN行だけを表示せよ．確認にはheadコマンドを用いよ.

import pandas as pd
import sys

def extract_head_rows(n_rows, path):
    df = pd.read_table(path, header=None)
    return df.iloc[ :n_rows , : ]

if __name__ == "__main__":
    while True:
        try:
            n_rows = int(input("Number of rows (head): "))
            break
        except ValueError:
            print("Please enter a valid number")

    df = extract_head_rows(n_rows, "data/popular-names.txt")
    print(df)
