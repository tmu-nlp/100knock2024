import urllib.request
#withを使うことでcloneしなくてもよい
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    lines = f.readlines()
    count = len(lines)
    print(count)
#lines はリスト