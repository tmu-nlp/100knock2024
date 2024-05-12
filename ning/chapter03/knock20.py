import gzip
import json

file_path = 'jawiki-country.json.gz'

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            print(article['text'])
            break
