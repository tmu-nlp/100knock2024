#ファイルをN分割する
#自然数Nをコマンドライン引数などの手段で受け取り，入力のファイルを行単位でN分割せよ．
#同様の処理をsplitコマンドで実現せよ．

n = 2
idx = df.shape[0] // n
for i in range(n):
    df_split = df.iloc[i * idx:(i + 1) * idx]
    df_split.to_csv(f"popular-names{i}.txt", sep="\t",header=False, index=False)

#UNIXコマンドで確認
!split -n 2 popular-names.txt
