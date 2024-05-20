#問25
#記事中に含まれる「基礎情報」テンプレートのフィールド名と値を抽出し，辞書オブジェクトとして格納せよ．


#方針１：空のdicを作成し、イギリスデータの要素を改行文字で区切る
#方針２：「基本情報」テンプレートのフィールド名と値の形に注目して、正規表現のメタ文字を記述
#if re.search(“|(.+?)\s=\s*(.+)”, text):
#でTrueになった値にたいしてキーと値を指定し、dictに追加
#基礎情報テンプレートの「|フィールド名 ＝ 値」で記述されている箇所を抽出

import pandas as pd
import re 

filename = "jawiki-country.json"

j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

#re.search(パターン, 対象文字列)メソッド：インデックスを指定することでこのmatchオブジェクトから3種類の値を取得できる
#                     インデックス0→マッチしたオブジェクトが存在する行、
#　　　　　　　　　　　　　　　　 1→マッチしたオブジェクトだけ
#                                 2→マッチしたオブジェクト以外の値を返す。
#                     戻り値は、match型。
#'"(.*)"'：ダブルクォーテーションで囲まれた文字列
#\s：任意の空白文字
dic={}
for text in uk_df[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s*=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]
print(dic)
