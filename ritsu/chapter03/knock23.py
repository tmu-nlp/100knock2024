import re
from knock20 import find_uk_article

def extract_sections(text):
    """記事テキストからセクション名とそのレベルを抽出する"""
    pattern = r'(={2,})\s*(.*?)\s*\1' # セクション名を表す正規表現, ={2,}は2回以上の=にマッチ, \s*は0回以上の空白文字にマッチ, .*?は可能な限り短い文字列にマッチ, \1は1番目のキャプチャグループと同じ文字列にマッチ
    results = re.findall(pattern, text) # findallでマッチする部分を全て取得, タプルのリストで返す, 例: [(==, 見出し1), (===, 見出し2), ...]
    for result in results:
        level = len(result[0]) - 1  # 等号の数から1を引くことでレベルを決定
        section_name = result[1] # セクション名, result[1]にマッチした文字列が入っている
        print(f'レベル{level}: {section_name}')

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        extract_sections(uk_text)
    else:
        print("「イギリス」の記事が見つかりませんでした。")
