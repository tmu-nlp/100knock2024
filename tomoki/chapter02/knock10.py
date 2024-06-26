#NO10(行数のカウント)
import pandas as pd
#df=pd.read_table("/home/numauo/100knock2024/tomoki/chapter01/popular-names.txt")
df = pd.read_table("popular-names.txt",header=None)
print(df.shape[0])

#wc -l "popular-names.txt"