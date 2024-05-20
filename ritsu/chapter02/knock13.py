def merge_columns(col1_file, col2_file, output_file):
    with open(col1_file, "r") as file1, open(col2_file, "r") as file2, open(output_file, "w") as output:
        for col1_line, col2_line in zip(file1, file2): #zip関数でcol1.txtとcol2.txtを同時に読み込む
            col1_content = col1_line.strip()  # 改行を削除
            col2_content = col2_line.strip()  # 改行を削除
            output.write(f"{col1_content}\t{col2_content}\n")  # タブで区切ってファイルに書き込む

if __name__ == "__main__":
    merge_columns("col1.txt", "col2.txt", "merged.txt")

# # pasteコマンドは、複数のファイルを行方向に結合するコマンド, defaultではタブで区切る
# paste col1.txt col2.txt
