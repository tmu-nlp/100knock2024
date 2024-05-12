import pandas as pd
basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"

with open(basepath+"col1.txt","r") as f:
    lines = f.read().splitlines()
f.close()

df_names = pd.DataFrame({"names": lines, "count": len(lines) * [1]})#それぞれの名前にcount1を追加する
df_count = df_names.groupby("names").sum()["count"]#名前ごとにcountの合計値を求める
df_count_renamed = df_count.rename("count").reset_index() # 列名"count"をつけて再度dataframe作成
print(df_count_renamed.sort_values("count",ascending=False))