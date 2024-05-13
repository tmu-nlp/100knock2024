#問16
#自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ
#同様の処理をsplitコマンドで実現せよ．

#DataFrameの行数をN分割数で割る（小数点切り捨て）
#df.iloc[i * idx:(i + 1) * idx]を指定して、DataFrameの行を分割
#csvファイルにデータをタブ区切りで書き込む

import pandas as pd
import subprocess

df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

n = int(input())
idx = df.shape[0] // n  #データ数を分割数Nで割る（切り捨て除算）
for i in range(n):
    df_split = df.iloc[i * idx:(i + 1) * idx]  #i * idxから(i + 1) * idxの一つ手前までのDataFrameの行を抽出している。
    df_split.to_csv(f"popular-names{i}.txt", sep="\t",header=False, index=False)

#UNIXコマンド
command = ["split", "-n num /home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt"]
subprocess.check_output(command) 
