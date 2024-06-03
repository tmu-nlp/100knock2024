import re
#ファイル:…がウィキペディアのマークアップ言語で画像を示している
#[[ファイル: で始まり、| または ] で終わる文字列を検索
pattern = '\[\[ファイル:(.*?)(?:\||\])'
result = re.findall(pattern, UK_text)
result