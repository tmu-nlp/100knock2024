import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
new = df[0].unique()
new.sort()
print (new)