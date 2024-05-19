import re
from knock20 import find_uk_article

def extract_category_lines(text):
    """記事テキストからカテゴリ行を抽出する"""
    pattern = r'\[\[Category:[^\]]+\]\]' # カテゴリ行を表す正規表現, [^\]]は]以外の文字にマッチ
    return re.findall(pattern, text) # findallでpatternにマッチする部分を全て取得

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        category_lines = extract_category_lines(uk_text)
        for line in category_lines:
            print(line)
    else:
        print("「イギリス」の記事が見つかりませんでした。")
