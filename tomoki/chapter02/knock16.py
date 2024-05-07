#NO16(ファイルをN分割する)
import sys
import pandas as pd
df=pd.read_table("popular-names.txt",header=None)
n=int(sys.argv[1])
#行数をｎで分割する
idx = df.shape[0] // n
#1回目で０~idx-1行目まで,2回目でidx~2idx-1行目まで、というようにファイルが作成されていく。
for i in range(n):
    df_split = df.iloc[i * idx:(i + 1) * idx]
    df_split.to_csv(f"popular-names{i}.txt", sep="\t",header=False, index=False)