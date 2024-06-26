import re
from knock27 import process_text_with_links_removed, find_uk_article  # knock27から必要な関数をインポート

def remove_extra_markup(text):
    """MediaWikiの追加マークアップを除去する"""
    # HTMLタグの除去
    text = re.sub(r'<[^>]+>', '', text) # <[^>]+>は<と>で囲まれた文字列にマッチ
    # テンプレート引数の除去
    text = re.sub(r'\{\{[^{]+?\}\}', '', text) # \{\{[^{]+?\}\}は{{と}}で囲まれた文字列にマッチ
    # 外部リンクの除去
    text = re.sub(r'\[http[^\s]+?\s([^\]]+?)\]', r'\1', text) # \[http[^\s]+?\s([^\]]+?)\]は[http://... ...]の形式にマッチ
    # コメントの除去
    text = re.sub(r'<!--.*?-->', '', text) # <!--.*?-->は<!--と-->で囲まれた文字列にマッチ
    # リファレンス/参照の除去
    text = re.sub(r'<ref[^>]*>(.*?)<\/ref>', '', text) # <ref[^>]*>(.*?)<\/ref>は<ref>と</ref>で囲まれた文字列にマッチ
    text = re.sub(r'<ref[^/>]*/>', '', text) # <ref[^/>]*/>は<ref/>にマッチ
    return text

def process_cleaned_text(text):
    """テキストデータを完全にクリーンアップし、辞書を返す"""
    info_dict = process_text_with_links_removed(text)
    fully_cleaned_dict = {key: remove_extra_markup(value) for key, value in info_dict.items()}
    return fully_cleaned_dict

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        fully_cleaned_info = process_cleaned_text(uk_text)
        for key, value in fully_cleaned_info.items():
            print(f"{key}: {value}")
    else:
        print("「イギリス」の記事が見つかりませんでした。")
