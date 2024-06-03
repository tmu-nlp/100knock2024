def save_columns(input_file):
    with open(input_file, "r") as file:
        col1 = []
        col2 = []
        for line in file:
            columns = line.strip().split('\t')  # 行からタブを削除し、タブで分割してリストにする
            col1.append(columns[0]) # 1列目をリストに追加
            col2.append(columns[1]) # 2列目をリストに追加
    
    with open("col1.txt", "w") as file1:
        file1.write("\n".join(col1) + "\n")  # 1列目のデータをファイルに書き出し

    with open("col2.txt", "w") as file2:
        file2.write("\n".join(col2) + "\n")  # 2列目のデータをファイルに書き出し

if __name__ == "__main__":
    save_columns("popular-names.txt")

# # cutコマンドで1列目を抽出, -f 1は1列目を指定
# cut -f 1 popular-names.txt

# # cutコマンドで2列目を抽出, -f 2は2列目を指定,
# cut -f 2 popular-names.txt
