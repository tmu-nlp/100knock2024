import urllib.request
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f, open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col1.txt","w") as f1 , open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col2.txt","w") as f2: #書き出し用のファイル設定
    lines = f.readlines()
    for line in lines:
        col1 = line.decode().split()[0]
        f1.write(col1+"\n")
        col2 = line.decode().split()[1]
        f2.write(col2+"\n")