import pandas as pd
number = int(input('出力したい先頭の行数を入力してください：'))
df = pd.read_table('./popular-names.txt', header=None)
print (df.head(number))