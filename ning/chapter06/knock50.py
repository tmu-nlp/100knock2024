"""
News Aggregator Data Setを使い、
学習データ（train.txt），検証データ（valid.txt），評価データ（test.txt）を作成するためには
１、データの読み込み
２、”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”の事例（記事）のみを抽出する
３、抽出したデータをランダムに並び替える
４、80%を学習データ、10%ずつを検証データと評価データに分割
５、train.txt，valid.txt，test.txtで,１行に１事例を書き出すこととカテゴリ名と記事見出しのタブ区切り形式に保存
６、カテゴリごとの事例数を確認
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# データの読み込み
data = pd.read_csv('newsCorpora.csv', sep='\t', header=None)
#　コラムがないため、header指定しないことにより、データの最初の行が列名として使わず、すべての行がデータとして読み込まれる
#　列名の指定
data.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']

# 指定された事例のみを抽出
publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
filtered_data = data[data['PUBLISHER'].isin(publishers)]

# データのシャッフル
# fracはサンプリングするデータの割合を指定する引数で、1を指定すると全ての行が対象になる
# random_stateは乱数生成器のシード値を設定する引数
# .reset_index()で元のインデックスはデフォルトで新しい列として追加
# drop=Trueは元のインデックス列を削除するオプションで、新しいインデックスだけが残る
shuffled_data = filtered_data.sample(frac=1, random_state=42).reset_index(drop=True)

# データの分割
# test_size=0.2は80%のデータを抽出することを指定している、20%はtempに入れる
# vaildとtestにtempの半分筒入れる
# random_stateを上と同じように指定すること
train, temp = train_test_split(shuffled_data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# 必要な列（カテゴリ名と記事見出し）のみを保存
train_data = train[['CATEGORY', 'TITLE']]
valid_data = valid[['CATEGORY', 'TITLE']]
test_data = test[['CATEGORY', 'TITLE']]

# ファイルへの保存
# カテゴリと見出しがタブ区切り形式で保存
train_data.to_csv('train.txt', sep='\t', index=False, header=False)
valid_data.to_csv('valid.txt', sep='\t', index=False, header=False)
test_data.to_csv('test.txt', sep='\t', index=False, header=False)

# カテゴリごとの事例数を確認
train_counts = train['CATEGORY'].value_counts()
valid_counts = valid['CATEGORY'].value_counts()
test_counts = test['CATEGORY'].value_counts()

print("Training Data Counts:\n", train_counts)
print("Validation Data Counts:\n", valid_counts)
print("Test Data Counts:\n", test_counts)

