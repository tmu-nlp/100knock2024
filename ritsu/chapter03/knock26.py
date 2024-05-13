import re
from knock25 import extract_basic_info, find_uk_article  # knock25から必要な関数をインポート

def remove_markup(text):
    """MediaWikiのマークアップを除去する"""
    # 強調マークアップの除去
    text = re.sub(r"''+", '', text)  # 二つ以上のシングルクォートを削除, subは置換
    return text

def process_text(text):
    """テキストデータを処理して、マークアップを除去した辞書を返す"""
    info_dict = extract_basic_info(text)
    cleaned_dict = {key: remove_markup(value) for key, value in info_dict.items()} # 辞書内包表記で各値にremove_markupを適用
    return cleaned_dict

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        cleaned_info = process_text(uk_text)
        for key, value in cleaned_info.items():
            print(f"{key}: {value}")
    else:
        print("「イギリス」の記事が見つかりませんでした。")
