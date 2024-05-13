import urllib.request
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    content = f.read().decode("utf-8")
    print(content.replace('\t',' '))