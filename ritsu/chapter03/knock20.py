import json, gzip

def find_uk_article(filename):
    """「イギリス」に関する記事を取得する"""
    with gzip.open(filename, 'r') as f:
        for line in f:
            article = json.loads(line)
            if article['title'] == 'イギリス':
                return article['text']
    return None

if __name__ == "__main__":
    filename = 'jawiki-country.json.gz'
    uk_text = find_uk_article(filename)
    if uk_text:
        print(uk_text)
        with open('uk.txt', 'w', encoding='utf-8') as file:
            file.write(uk_text)
    else:
        print("「イギリス」の記事が見つかりませんでした。")