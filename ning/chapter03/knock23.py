import gzip
import json
import re

file_path = 'jawiki-country.json.gz'

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            text = article['text'].split('\n')
            for line in text:
                match = re.match(r'^(=+)([^=]+)=+$', line)
                if match:
                    level = len(match.group(1)) - 1 
                    section_name = match.group(2).strip() 
                    print(f"レベル{level}: {section_name}")
