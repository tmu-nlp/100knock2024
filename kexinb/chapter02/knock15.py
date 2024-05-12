# task 15. 末尾のN行を出力
# 自然数Nをコマンドライン引数などの手段で受け取り，入力のうち末尾のN行だけを表示せよ．確認にはtailコマンドを用いよ．

import pandas as pd

def extract_tail_rows(n_rows, path):
    df = pd.read_table(path, header=None)
    return df.iloc[-n_rows: , : ]

if __name__ == "__main__":
    while True:
        try:
            n_rows = int(input("Number of rows (tail): "))
            break
        except ValueError:
            print("Please enter a valid number")

    out = extract_tail_rows(n_rows, "data/popular-names.txt")
    print(out)