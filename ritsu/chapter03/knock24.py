import re
from knock20 import find_uk_article

def extract_files(text):
    """記事テキストからメディアファイルのリンクを抽出する"""
    pattern = r'\[\[(ファイル|File):(.*?)(\||\]\])' # メディアファイルのリンクを表す正規表現, (.*?)により:のあとの可能な限り短い文字列にマッチする
    files = re.findall(pattern, text)
    return [file[1] for file in files]  # ファイル名の部分だけを返す

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        files = extract_files(uk_text)
        for file in files:
            print(file)
    else:
        print("「イギリス」の記事が見つかりませんでした。")
