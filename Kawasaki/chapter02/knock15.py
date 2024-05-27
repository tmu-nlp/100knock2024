import pandas as pd
number = int(input('出力したい最後の行数を入力してください：'))
df = pd.read_table('./popular-names.txt', header=None)
print (df.tail(number))