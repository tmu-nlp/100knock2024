#NO14(先頭からN行を出力)
import sys
import pandas as pd
df=pd.read_table("popular-names.txt",header=None)
#受け渡し時、型に注意
value = sys.argv[1]
print(df.head(int(value)))

#python knock14.py 5

#行数は-nを使用する。
#head -n 5 popular-names.txt