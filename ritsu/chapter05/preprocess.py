import zipfile
import CaboCha

# ai.ja.zipを解凍してテキストを取得
with zipfile.ZipFile('ai.ja.zip', 'r') as zip_ref:
    text = zip_ref.read('ai.ja.txt').decode('utf-8')

# CaboChaを使って係り受け解析を実行
cabocha = CaboCha.Parser()
parsed_text = cabocha.parse(text).toString(CaboCha.FORMAT_LATTICE)

# 解析結果をファイルに保存
with open('ai.ja.txt.parsed', 'w', encoding='utf-8') as f:
    f.write(parsed_text)