#23.セクション構造
#記事中に含まれるセクション名とそのレベル
#（例えば”== セクション名 ==”なら1）を表示せよ．

for text in uk_df[0].split("\n"):
    if re.search("^=+.*=+$", text):
        num = text.count("=") / 2 - 1
        print(text.replace("=", ""), int(num))