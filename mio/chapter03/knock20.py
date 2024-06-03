#問20
#Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．
#問題21-29では，ここで抽出した記事本文に対して実行せよ．

#jsonファイルを読み込み、DataFrameに変換
#DataFrameの”title”列データが”イギリス”である行だけを取得
#uk_df[“text”]列のデータを配列（numpy.ndarray）に変換し出力

import pandas as pd

#pd.read_json()関数：第一引数に、Json形式の文字列を渡すことで、文字列がpd.DataFrameに変換される/JSON Linesで書かれたファイルを読み込みたい場合は、lines=Trueを指定する。
filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data

#DataFrameに比較演算子を使用する：データをbool型として出力
#　　　　　　　　　　　　　　　　 Tureとなる場所をDataFrameに指定することで、その行のデータを取得可能
uk_df = df[df["title"]=="イギリス"]

#DataFrame.values属性（Series.values)：pandasのDataFrameとSeriesオブジェクトに対して、values属性を指定すると、NumPy配列(ndarry)に変換できる
uk_df = uk_df["text"].values
print(uk_df)

