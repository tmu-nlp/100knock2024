import urllib.request
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    N = int(input())
    lines = f.readlines()
    #行バイト型でリスト型として入っている
    for i in range(N):
        #i行目を取り出し、その行をバイト型から文字列型に変換
        l = lines[i].decode()
        print(l)
        