#10.行数のカウント
#行数をカウントせよ．確認にはwcコマンドを用いよ．

import pandas as pd
import subprocess

path_name="/Users/megumi/python_megumi/100knock2024/megumi/chapter02/"
df=pd.read_table(path_name+"popular-names.txt", header=None)
print(len(df))


