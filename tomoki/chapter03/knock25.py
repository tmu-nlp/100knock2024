#No25(テンプレートの抽出)
#フィールド名　それぞれのデータ項目に付ける名称　ex顧客の名前の場合「名前」　「氏名」など
import pandas as pd
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
dic={}
for text in D_8[0].split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s*=\s*(.+)", text)
        #インデックスが1ならマッチしたオブジェクトだけ、2ならマッチしたオブジェクト以外の値を返す。
        dic[match_txt[1]] = match_txt[2]
print(dic)


#「\s」 空白
#|で始まり、何か文字が入って、空白が入る（但し、必ずしも空白が入るわけではないため\s*としている）。
#その後「＝」と空白が入り、何か文字が入るというパターンをマッチングするようにする。

#{{基礎情報 国\n|略名  =イギリス\n|日本語国名 = グレートブリテン及び北アイルランド連合王国\n|公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />\n*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（