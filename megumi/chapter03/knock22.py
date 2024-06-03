#22.カテゴリ名の抽出
#記事のカテゴリ名を（行単位ではなく名前で）抽出せよ．

for text in uk_df[0].split("\n"):
    if re.search("Category", text):
        text = text.replace("[[Category:", "").replace("|*]]", "").replace("]]", "")
        print(text)