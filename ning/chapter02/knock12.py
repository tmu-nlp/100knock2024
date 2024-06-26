with open("popular-names.txt", "r") as f, open("col1.txt", "w") as fc1, open("col2.txt", "w") as fc2:#読み取りと書き込み
    lines = f.readlines()#すべての行を読み取り、リストとして返す
    count = len(lines)
    for i, line in enumerate(lines): #リストの各要素とそのインデックスを取得
        parts = line.split()
        name = parts[0]
        sex = parts[1]
        if i == count - 1:  #最後の行の処理
            fc1.write(name)  #最後の行では改行を加えない
            fc2.write(sex)
        else:#改行を追加
            fc1.write(name + '\n') 
            fc2.write(sex + '\n') 

#1列目
#cut -f 1 -d " " '[PATH]/popular-names.txt'
#2列目
#cut -f 2 -d " " '[PATH]/popular-names.txt'

