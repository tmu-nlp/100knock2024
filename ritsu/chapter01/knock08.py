def cipher(text):
    """
    与えられたテキストを暗号化/復号化。英小文字は(219 - 文字コード)に置換、その他の文字はそのまま出力。
    ord(char) は文字 char のASCIIコードを返す
    chr(...)はASCIIコードを文字に変換する
    (ASCIIコードとは...アスキーコード。標準体な数値と文字の対応付け)
    """
    return ''.join(chr(219 - ord(char)) if 'a' <= char <= 'z' else char for char in text)

# 使用例
original_text = "Hello, World! This is a test message with ONLY lowercase and UPPERCASE."
encrypted_text = cipher(original_text)
decrypted_text = cipher(encrypted_text)

# print("Original:", original_text)
# print("Encrypted:", encrypted_text)
# print("Decrypted:", decrypted_text)