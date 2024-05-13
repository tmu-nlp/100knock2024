def replace_tabs_with_spaces(text_path, output_path):
    # ファイルを開き、タブをスペースに置換して別のファイルに保存する
    with open(text_path, "r") as input_file:
        with open(output_path, "w") as output_file:
            for line in input_file:
                output_file.write(line.replace('\t', ' '))  # タブをスペースに置換して出力

if __name__ == "__main__":
    replace_tabs_with_spaces("popular-names.txt", "output_11.txt")  # 出力結果はoutput_11.txtに保存

# # sedコマンド. 's/\t/ /g' という式で、タブ(\t)をスペースに置換(s)し、全ての出現箇所(g)に対して適用する。
# sed 's/\t/ /g' popular-names.txt

# # trコマンド. タブ(\t)をスペースに置換する。<は入力ファイルを指定するリダイレクト記号。
# tr '\t' ' ' < popular-names.txt

# # expandコマンド. タブ(\t)をスペースに置換する。-t 1はタブの幅を1に指定している。
# expand -t 1 popular-names.txt