import pandas as pd
basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"

df_names = pd.read_table(basepath+"popular-names.txt",names=["名前","性別","人数","年"])
print(df_names.sort_values("人数",ascending=False))
