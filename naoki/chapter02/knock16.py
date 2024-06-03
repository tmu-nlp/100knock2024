import urllib
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    N = int(input())
    #linesの要素がバイト型なので文字列に変換/.decodeはリスト全体に対して作用させることが出来ないので、このような表記を用いた
    lines = [line.decode('utf-8') for line in f.readlines()]
    count = len(lines) // N

    for i in range(N):
        if i == N-1:
            part_lines = lines[i*count:]
        else:
            part_lines = lines[i*count:(i+1)*count]
            with open(f'C:/Users/shish_sf301y1/Desktop/pyファイル/split{i+1}.txt','w') as file: 
                file.writelines(part_lines)