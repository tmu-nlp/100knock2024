import urllib.request
#withを使うことでcloneしなくてもよい
#closeし忘れると、リソースを食ったり、書き込みが正常に行われないなどが起こる
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    lines = f.readlines()
    count = len(lines)
    print(count)
#lines は1文のリスト

"""
UNIXコマンド
wc -l "popular-names.txt"
"""