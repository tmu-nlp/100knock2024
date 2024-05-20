#21.カテゴリ名を含む行を抽出
#記事中でカテゴリ名を宣言している行を抽出せよ．

import re

#uk_dfの要素数を確認
#print(len(uk_df))
#uk_dfのデータ型を確認
#print(type(uk_df[0]), "\n")

for text in uk_df[0].split("\n"):
    if re.search("Category", text):
        print(text)