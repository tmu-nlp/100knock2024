#23は全く分からない
import gzip
import json
import re

# gzip圧縮されたファイルを開く
with gzip.open('jawiki-country.json.gz', 'rt', encoding='utf-8') as file:
    # ファイル内の全ての行を読み込む
    for line in file:
        article = json.loads(line)
        text = article['text']
        # セクション名を抽出する正規表現パターン
        pattern = r'(==+)([^=]+)\1'
        # テキスト内の全てのセクション名を検索
        sections = re.findall(pattern, text)
        for section in sections:
            # セクション名の前後にある'='の数でレベルを決定
            level = len(section[0]) - 1
            # セクション名とそのレベルを表示
            print(f'セクション名: {section[1].strip()}, レベル: {level}')