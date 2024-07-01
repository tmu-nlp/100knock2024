import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split

#データに名前づけ
header_name = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
df = pd.read_csv('newsCorpora.csv', header=None, sep='\t', names=header_name)
"""
情報源（publisher）が
”Reuters”, “Huffington Post”, “Businessweek”, “Contactmusic.com”, “Daily Mail”
の事例（記事）のみを抽出
"""
#[]の使い分けがよくわからない
df_new = df[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]

#データの分割
df_train, df_other = train_test_split(
    df_new, test_size=0.2, random_state=777)
df_valid, df_test = train_test_split(
    df_other, test_size=0.5, random_state=777)

#CSV形式で書き出し
df_train.to_csv('df_train.txt',sep='\t', index=False,header=False)
df_valid.to_csv('df_valid.txt',sep='\t', index=False,header=False)
df_test.to_csv('df_test.txt',sep='\t', index=False,header=False)

#カテゴリーの事例数を確認 .value_counts()はあるデータカラムにあるタイプごとの総数がわかるので便利
print(df_train['CATEGORY'].value_counts())
print(df_valid['CATEGORY'].value_counts())
print(df_test['CATEGORY'].value_counts())