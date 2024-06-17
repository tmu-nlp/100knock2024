import pickle
from scipy.stats import spearmanr

with open("drive/MyDrive/word2vec.pkl", "rb") as f:
    model = pickle.load(f)
word1 = []
word2 = []
human_rank = []
model_rank = []

df = pd.read_csv("drive/MyDrive/combined.csv",header=0)
df = df.dropna()
for i in range(len(df)):
    word1.append(df.iloc[i,0])
    word2.append(df.iloc[i,1])
    human_rank.append(df.iloc[i,2])
    cos = model.similarity(df.iloc[i,0],df.iloc[i,1])
    model_rank.append(cos)
#model_rankにあるデータを順位データに変換
model_rank = pd.Series(model_rank).rank(ascending=True, method='min')
correlation, pvalue = spearmanr(human_rank,model_rank)
print(correlation)
print(pvalue)