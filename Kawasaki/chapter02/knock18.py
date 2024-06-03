import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
new = df[2].sort_values(ascending=False)
print (new)