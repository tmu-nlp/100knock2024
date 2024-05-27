import re
from knock20 import find_uk_article

def extract_category_names(text):
    """記事テキストからカテゴリ名だけを抽出する"""
    # 正規表現を用いてカテゴリ名を抽出
    pattern = r'\[\[Category:(.*?)(?:\||\]\])' # (.*?)により:のあとの可能な限り短い文字列にマッチする, ?: はキャプチャしないことを意味
    return re.findall(pattern, text)

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        category_names = extract_category_names(uk_text)
        for name in category_names:
            print(name)
    else:
        print("「イギリス」の記事が見つかりませんでした。")
