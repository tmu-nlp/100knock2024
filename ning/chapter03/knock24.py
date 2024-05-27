import gzip
import json
import re

file_path = 'jawiki-country.json.gz'

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            text = article['text'].split('\n') 
            files = []
            for line in text:
                matches = re.findall(r'\[\[(ファイル|File):([^|\]]+)', line)
                files.extend([match[1].strip() for match in matches])
            print(files)
            break
