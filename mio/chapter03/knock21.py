#問21
#記事中でカテゴリ名を宣言している行を抽出せよ．

#reをimportする。
#問20で取得したデータの要素数とデータ型を確認（コメント）
#uk_dfの要素を改行文字で区切る
#カテゴリ名を宣言している行、つまりCategoryから始まる行を出力
import pandas as pd
import re #re標準モジュール：正規表現操作を行うモジュール

#pd.read_json()関数：第一引数に、Json形式の文字列を渡すことで、文字列がpd.DataFrameに変換される/JSON Linesで書かれたファイルを読み込みたい場合は、lines=Trueを指定する。
filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

#uk_dfの要素数を確認
#print(len(uk_df))

#uk_dfのデータ型を確認
#print(type(uk_df[0]), "\n")


#str.split()メソッド：引数で指定した文字列でstrオブジェクトを分割
#　　　　　　　　　　 戻り値は、list型xd
#　　　　　　　　　　 デフォルトで改行文字で区切る(”\n”を指定しなくてもいい)
#re.search()関数：第1引数は、文字列またはstr型のメタ文字。第2引数は、検索対象となるstrオブジェクトを指定する。戻り値は、match型。マッチする文字列が複数あった場合でも、最初にマッチする文字列しか返さないことに注意する。
#uk_dfの要素数は1、データ型はstr。
for text in uk_df[0].split("\n"):
    if re.search("Category", text):
        print(text)