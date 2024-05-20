import requests
from knock28 import process_cleaned_text, find_uk_article  # knock28から必要な関数をインポート

def fetch_image_url(filename):
    """指定された画像ファイル名に対応するURLをMediaWiki APIから取得する"""
    endpoint = "https://commons.wikimedia.org/w/api.php" # APIのURL
    params = {
        "action": "query", # クエリを実行する
        "format": "json", # JSON形式で出力
        "prop": "imageinfo", # 画像情報を取得
        "titles": f"File:{filename}", # ファイル名を指定
        "iiprop": "url" # 画像のURLを取得
    }
    response = requests.get(endpoint, params=params).json() # APIにリクエストを送信
    page = next(iter(response['query']['pages'].values()), {}) # ページ情報を取得, nextで最初の要素を取得
    return page.get('imageinfo', [{}])[0].get('url', "URL not found") # 画像のURLを取得, [{}]は空のリストを指定, getでキーが存在しない場合のデフォルト値を指定

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        info = process_cleaned_text(uk_text)
        flag_filename = info.get('国旗画像', '').replace(' ', '_') # 国旗画像のファイル名を取得, スペースをアンダースコアに置換することでURLを生成
        print(f"URL for the national flag image: {fetch_image_url(flag_filename)}")
    else:
        print("「イギリス」の記事が見つかりませんでした。")

