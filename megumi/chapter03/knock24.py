#24.ファイル参照の抽出
#記事から参照されているメディアファイルをすべて抜き出せ．


m_file = re.findall("ファイル:(.+?)\|", uk_df[0])
print("\n".join(m_file))