#NO12(1列目をcol1.txtに，2列目をcol2.txtに保存)
import pandas as pd
df=pd.read_table("popular-names.txt",header=None)
#iloc[行,列]
df.iloc[:,0].to_csv("col1.txt",index=None,header=None)
df.iloc[:,1].to_csv("col2.txt",index=None,header=None)

#unixコマンド
#cut -f 1 popular-names.txt > col1u.txt
#cut -f 2 popular-names.txt > col2u.txt

#loc 行名もしくは列名を指定することで特定の値を抽出

