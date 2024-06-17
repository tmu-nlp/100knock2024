import urllib.request
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f, open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col1.txt","w") as f1 , open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col2.txt","w") as f2: #書き出し用のファイル設定
    lines = f.readlines()
    for line in lines:
        col1 = line.decode().split()[0]
        f1.write(col1+"\n")
        col2 = line.decode().split()[1]
        f2.write(col2+"\n")
"""
UNIXコマンド
cut -f 1 popular-names.txt
cut -f 2 popular-names.txt
"""

"""
encode 文字列⇒バイト列
decode バイト列⇒文字列
バイト列:
「１バイトのデータを並べた、データの集まり」のことです。
（例）b'\xe3\x81\x82'
左端の「b」は、「バイト列」を意味する記号です。
その隣の「xe3」の「x」は、「16進数形式」を意味する記号です。
数字「e3」が、16進数形式で表示された「1バイト（8ビット）」のデータです。
"""