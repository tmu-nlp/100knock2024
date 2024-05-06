import pandas as pd
basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
uk_list=[]#後で出力する

df = pd.read_json(basepath+"jawiki-country.json",lines=True)
#json lines:改行区切りで，1行に1jsonオブジェクトがある形式のファイル．->lines=True
#lines=FAlseにすると　->Trailing dataでエラーになる．

for i in range(len(df)):
    if df["title"][i] == "イギリス":
        print(df["text"][i])
        uk_list.append(df["text"][i])

uk_df = pd.DataFrame(uk_list)
uk_df.to_csv(basepath+"uk_articles.txt",encoding='utf-8_sig')
#encodingしないと文字化けした