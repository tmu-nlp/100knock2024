#問26
#25の処理時に，テンプレートの値からMediaWikiの強調マークアップ（弱い強調，強調，強い強調のすべて）を除去してテキストに変換せよ

#参考：https://leadinge.co.jp/rd/2021/11/22/1501/#toc8
#正規表現について：https://userweb.mnet.ne.jp/nakama/

#re.sub(正規表現パターン, 置換先文字列, 処理対象の文字列)メソッド
#：第２引数\\1：第一引数のメタ文字の一部を()で囲むと、マッチしたオブジェクトを（）内の文字列で置換

#正規表現
#\'：シングルクオート文字そのもの（'）（シングルクオート文字列内でシングルクオート文字を使うとき）
#{2,}：直前の文字が２桁以上
#.+? ：任意の1文字以上の文字
#　　　　 .：改行以外の任意の文字  
#　　　　 +：直前の文字の1回以上の繰り返し   
#         ?：直前のパターンを0または1回繰り返したもの	

import pandas as pd
import re 

filename = "jawiki-country.json"

#lines：JSONオブジェクトが1行ずつ書かれているとしてファイルを読み込む。デフォルトはFalse。
j_data = pd.read_json(filename, lines =True)
df = j_data
uk_df = df[df["title"]=="イギリス"]

dic = {}
for text in uk_df["text"].values.split("\n"):
    if re.search("\|(.+?)\s=\s*(.+)", text):
        match_txt = re.search("\|(.+?)\s=\s*(.+)", text)
        dic[match_txt[1]] = match_txt[2]      
    match_sub = re.sub("\'{2,}(.+?)\'{2,}", "\\1", text)
    print(match_sub)

