import re
from knock26 import remove_markup, process_text, find_uk_article  # knock26から必要な関数をインポート

def remove_internal_links(text):
    """MediaWikiの内部リンクマークアップを除去する"""
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^|\]]+)\]\]', r'\1', text)  # 通常の内部リンクの形式を処理, \1は1番目のキャプチャグループにマッチした文字列に置換
    # 国章画像のファイルの除去
    text = re.sub(r'\[\[ファイル:(?:[^|\]]*\|)*([^|\]]+)\]\]', r'\1', text)
    return text

def process_text_with_links_removed(text):
    """テキストデータを処理して、マークアップと内部リンクを除去した辞書を返す"""
    info_dict = process_text(text)  # knock26.pyのprocess_textを使用
    cleaned_dict = {key: remove_internal_links(remove_markup(value)) for key, value in info_dict.items()}
    return cleaned_dict

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        cleaned_info = process_text_with_links_removed(uk_text)
        for key, value in cleaned_info.items():
            print(f"{key}: {value}")
    else:
        print("「イギリス」の記事が見つかりませんでした。")

