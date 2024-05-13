import gzip
import json
import re 

file_path = 'jawiki-country.json.gz'

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            text = article['text'].split('\n') 
            categories = []
            for line in text:
                match = re.search(r'\[\[Category:(.*?)(?:\|.*)?\]\]', line)
                if match:
                    categories.append(match.group(1))
            print(categories)
            break
