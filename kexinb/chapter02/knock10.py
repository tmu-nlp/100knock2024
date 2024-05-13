# task10: 行数のカウント
# 行数をカウントせよ．確認にはwcコマンドを用いよ．

import pandas as pd
# import subprocess

def count_lines(path):
    df = pd.read_csv(path, sep='\t', header=None)
    return len(df)

if __name__ == "__main__":
    filePath = 'data/popular-names.txt'
    print(count_lines(filePath))