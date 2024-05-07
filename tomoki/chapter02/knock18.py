#No18(各行を3コラム目の数値の降順にソート)
import pandas as pd
df=pd.read_table('popular-names.txt', header=None)

#３コラム(列)を降順にしていく(入れる数字に注意)。
print(df.sort_values(2,ascending=False))

#-n(文字列を数字とみなす)　-r(降順にする)　-k(場所と並べ替え種別を指定する)
#sort  -k 3nr  popular-names.txt
