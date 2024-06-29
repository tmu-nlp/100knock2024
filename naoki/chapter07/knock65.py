import pandas as pd

df = pd.read_csv('drive/MyDrive/knock64.txt', sep=' ', header=None)
df.head()
#df[3]にはすでに用意された正解データが格納されている
#df[4]は今回推測した値
#今回用意した類似度は何の意味があったのか
print((df[3] == df[4]).sum() / len(df))