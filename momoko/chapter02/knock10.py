#データ準備
import pandas as pd
import subprocess

path_name="/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"
df_names = pd.read_table(path_name+"popular-names.txt",header=None)
print(len(df_names))