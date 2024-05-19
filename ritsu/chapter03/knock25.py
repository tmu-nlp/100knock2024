import re
from knock20 import find_uk_article

def extract_basic_info(text):
    """記事テキストから基礎情報テンプレートの内容を抽出し、辞書に格納する"""
    # 基礎情報テンプレートの全体を抽出
    pattern = r'{{基礎情報.*?^}}$'
    basic_info_block = re.search(pattern, text, re.DOTALL | re.MULTILINE) # re.DOTALLで.が改行にもマッチ, re.MULTILINEで^が各行の先頭にマッチ
    if basic_info_block:
        basic_info_text = basic_info_block.group(0)
        # 各フィールド名と値のペアを抽出
        field_pattern = r'\n\|([^=]+?)\s*=\s*(.*?)(?=\n\||\n$)' # \n\|は改行と|にマッチ, [^=]+?は=以外の文字が1文字以上にマッチ, \s*は0文字以上の空白文字にマッチ, (?=\n\||\n$)は次の改行と|または改行の直前にマッチ
        fields = re.findall(field_pattern, basic_info_text, re.DOTALL)
        # フィールドと値を辞書に格納
        info_dict = {field.strip(): value.strip() for field, value in fields}
        return info_dict
    return {}

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        info_dict = extract_basic_info(uk_text)
        for key, value in info_dict.items():
            print(f"{key}: {value}")
    else:
        print("「イギリス」の記事が見つかりませんでした。")




