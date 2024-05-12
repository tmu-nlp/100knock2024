basepath = "/Users/daining/Desktop/Python/100knock2024/"
with open(f"{basepath}col1.txt", "r") as fc1, open(f"{basepath}col2.txt", "r") as fc2, open(f"{basepath}merged.txt", "w") as f:
    linesfc1 = fc1.readlines()
    linesfc2 = fc2.readlines()
    count = min(len(linesfc1), len(linesfc2))  #col1.txtとcol2.txtの行数の最小値を取得
    for name, sex in zip(linesfc1, linesfc2):
        name = name.rstrip()  #改行文字を削除
        sex = sex.rstrip()    #改行文字を削除
        f.write(f"{name}\t{sex}\n")  #col1.txtとcol2.txtの対応する行をタブ区切りで結合して書き込み
