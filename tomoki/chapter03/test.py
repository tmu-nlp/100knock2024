import pandas as pd
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
print(D_8)