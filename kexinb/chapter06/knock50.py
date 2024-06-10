# task50. データの入手・整形
'''
The task is to classify a given news headline to one of the following categories: 
    “Business”, “Science”, “Entertainment” and “Health”
News Aggregator Data Setをダウンロードし、以下の要領で学習データ(train.txt), 検証データ(valid.txt),評価データ(test.txt)を作成せよ:
    ダウンロードしたzipファイルを解凍し,readme.txtの説明を読む.
    情報源(publisher)が”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, 
        “Daily Mail”の事例(記事)のみを抽出する.
    抽出された事例をランダムに並び替える.
    抽出された事例の80%を学習データ,残りの10%ずつを検証データと評価データに分割し,
        それぞれtrain.txt,valid.txt,test.txtというファイル名で保存する.ファイルには,１行に１事例を書き出すこととし,
        カテゴリ名と記事見出しのタブ区切り形式とせよ(このファイルは後に問題70で再利用する).
学習データと評価データを作成したら,各カテゴリの事例数を確認せよ.
'''
import pandas as pd
from sklearn.model_selection import train_test_split

header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
extract_list = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']

df = pd.read_csv('data/newsCorpora.csv', header=None, sep='\t', 
                 names=header_name)

df_ex = df.loc[df['PUBLISHER'].isin(extract_list), ['TITLE', 'CATEGORY']]

data_train, data_other = train_test_split(df_ex, test_size=0.2, random_state=42)
data_valid, data_test = train_test_split(data_other, test_size=0.5, random_state=42)

data_train.to_csv("output/ch6/train.txt", sep="\t", index=False, header=False)
data_valid.to_csv("output/ch6/valid.txt", sep="\t", index=False, header=False)
data_test.to_csv("output/ch6/test.txt", sep="\t", index=False, header=False)

if __name__ == "__main__":
    print("train_data")
    print(data_train['CATEGORY'].value_counts())
    print("valid_data")
    print(data_valid['CATEGORY'].value_counts())
    print("test_data")
    print(data_test['CATEGORY'].value_counts())

'''
train_data
CATEGORY
b    4538
e    4228
t    1205
m     701
Name: count, dtype: int64
valid_data
CATEGORY
b    531
e    529
t    155
m    119
Name: count, dtype: int64
test_data
CATEGORY
b    558
e    522
t    164
m     90
Name: count, dtype: int64
'''