#第2章: UNIXコマンド
#popular-names.txtは，アメリカで生まれた赤ちゃんの「名前」「性別」「人数」「年」をタブ区切り形式で格納したファイルである．
#以下の処理を行うプログラムを作成し，popular-names.txtを入力ファイルとして実行せよ．
#さらに，同様の処理をUNIXコマンドでも実行し，プログラムの実行結果を確認せよ．

#問10. 行数のカウント
#行数をカウントせよ．確認にはwcコマンドを用いよ．

import pandas as pd
import subprocess

#header引数：popular-names.txtはヘッダーがないので、Noneにしないとデータの1行目がヘッダーに指定されてしまう
df = pd.read_table("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt", header=None)

#DataFrame.shape属性(Series.shape属性)：DataFrame(Series)の次元をタプルで返す
#(今回のdataframeの戻り値は（行数, 列数）になっている→インデックスを指定すれば行数または列数を取得可能
print(df.shape[0])

#UNIXコマンド
command = ["wc", "/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt"]
subprocess.check_output(command) 
