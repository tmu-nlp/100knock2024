import gzip
import json
import re

file_path = 'jawiki-country.json.gz'

def extract_basic_info(text):
    pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}$', re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None

    contents = match.group(1)

    info_dict = {}
    pattern = re.compile(r'^\|\s*(.*?)\s*=\s*(.*?)\s*(?=\n\||\n\})', re.MULTILINE | re.DOTALL)
    fields = pattern.findall(contents)
    for field in fields:
        field_name = field[0]
        field_value = field[1]
        field_value = re.sub(r'\{\{.*?\}\}', '', field_value) 
        field_value = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', field_value) 
        field_value = re.sub(r'<.*?>', '', field_value) 
        field_value = re.sub(r"'''''(.*?)'''''", r'\1', field_value) 
        field_value = re.sub(r"'''(.*?)'''", r'\1', field_value)     
        field_value = re.sub(r"''(.*?)''", r'\1', field_value)         
        info_dict[field_name] = field_value.strip()

    return info_dict

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            basic_info = extract_basic_info(article['text'])
            print(basic_info)
            break
