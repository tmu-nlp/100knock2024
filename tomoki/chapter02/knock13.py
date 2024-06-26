#NO13(col1.txtとcol2.txtをマージ)
import pandas as pd
#header=Falseにしてはいけない
df1=pd.read_table("col1.txt",header=None)
df2=pd.read_table("col2.txt",header=None)
#axis=1は列方向に合体
df3=pd.concat([df1,df2],axis=1)
#sepで区切り文字(タブ区切り)を指定する
df3.to_csv("col_merge.txt", sep="\t", header=None, index=None)

#paste col1.txt col2.txt > col_merge2.txt