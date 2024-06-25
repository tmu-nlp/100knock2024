#50.データの入手・整形
"""
News Aggregator Data Setをダウンロードし、以下の要領で学習データ
（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成せよ．

1.ダウンロードしたzipファイルを解凍し，readme.txtの説明を読む．
2.情報源（publisher）が”Reuters”, “Huffington Post”, “Businessweek”, 
　“Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する．
3.抽出された事例をランダムに並び替える．
4.抽出された事例の80%を学習データ，残りの10%ずつを検証データと評価データに分割し，
　それぞれtrain.txt，valid.txt，test.txtというファイル名で保存する．
　ファイルには，１行に１事例を書き出すこととし，カテゴリ名と記事見出しのタブ区切り形式とせよ
　（このファイルは後に問題70で再利用する）1．

学習データと評価データを作成したら，各カテゴリの事例数を確認せよ．
"""

import pandas as pd
import numpy as np

# データの読み込み
df = pd.read_csv('newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# 指定された情報源の事例を抽出
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
df_filtered = df[df['PUBLISHER'].isin(publishers)]

# 抽出された事例をランダムに並び替える
df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

# データの分割
train_size = int(0.8 * len(df_filtered))
valid_size = int(0.1 * len(df_filtered))
test_size = len(df_filtered) - train_size - valid_size

train_data = df_filtered[:train_size]
valid_data = df_filtered[train_size:train_size + valid_size]
test_data = df_filtered[train_size + valid_size:]

# データの保存
train_data[['CATEGORY', 'TITLE']].to_csv('train.txt', sep='\t', index=False, header=False)
valid_data[['CATEGORY', 'TITLE']].to_csv('valid.txt', sep='\t', index=False, header=False)
test_data[['CATEGORY', 'TITLE']].to_csv('test.txt', sep='\t', index=False, header=False)

# 各カテゴリの事例数を確認
train_category_counts = train_data['CATEGORY'].value_counts()
valid_category_counts = valid_data['CATEGORY'].value_counts()
test_category_counts = test_data['CATEGORY'].value_counts()

print("Train category counts:\n", train_category_counts)
print("Valid category counts:\n", valid_category_counts)
print("Test category counts:\n", test_category_counts)


"""
出力結果
Train category counts:
 CATEGORY
b    4530
e    4178
t    1225
m     739
Name: count, dtype: int64
Valid category counts:
 CATEGORY
e    560
b    539
t    144
m     91
Name: count, dtype: int64
Test category counts:
 CATEGORY
b    558
e    541
t    155
m     80
Name: count, dtype: int64
"""