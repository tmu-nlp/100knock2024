#NO15(末尾のN行を出力)
import sys
import pandas as pd
df=pd.read_table("popular-names.txt",header=None)
#受け渡し時、型に注意
value = sys.argv[1]
print(df.tail(int(value)))

#python knock15.py 5

#tail -n 5 popular-names.txt