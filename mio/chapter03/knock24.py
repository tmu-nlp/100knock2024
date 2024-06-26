#問24
#記事から参照されているメディアファイルをすべて抜き出せ．

#ファイル参照の抽出

#方針１：イギリスデータを見ると、「ファイル：」の形でメディアファイルが記述されているので、findall(“ファイル:(.+?)|”, t_df[0])を使用する。
#方針２：最後m_fileリストの要素を改行文字で、連結させる。

import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]
#re.findall()：マッチするすべての文字列をlist型にして返す。リストの要素は、strオブジェクト（matchオブジェクトではない事に注意）
m_file = re.findall("ファイル:(.+?)\|", uk_df[0])
print("\n".join(m_file))