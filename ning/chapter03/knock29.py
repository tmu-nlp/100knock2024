import gzip
import json
import requests
import re

file_path = 'jawiki-country.json.gz'

def extract_flag_image(text):
    pattern = re.compile(r'\|\s*国旗画像\s*=\s*(.*?)\s*(?=\n)')
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None

def get_image_url(filename):
    S = requests.Session()
    URL = "https://commons.wikimedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{filename}",
        "iiprop": "url"
    }

    response = S.get(url=URL, params=PARAMS)
    data = response.json()
    pages = data['query']['pages']
    for k, v in pages.items():
        if 'imageinfo' in v:
            return v['imageinfo'][0]['url']
    return "Image not found"

with gzip.open(file_path, 'rt', encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        if article['title'] == 'イギリス':
            flag_image = extract_flag_image(article['text'])
            if flag_image:
                image_url = get_image_url(flag_image)
                print(f"Flag Image URL: {image_url}")
                break
