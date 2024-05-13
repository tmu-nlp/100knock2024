import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
vc = df[0].value_counts()
vc = pd.DataFrame(vc)
vc = vc.reset_index()
vc.columns = ['name','count']
vc = vc.sort_values(['count','name'],ascending=[False,False])
print (vc)