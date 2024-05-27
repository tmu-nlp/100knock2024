import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
df.to_csv('popular-names_replaced.txt', sep=' ',header=False, index=False)