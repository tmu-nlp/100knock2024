def count_lines(text_path):
    with open(text_path, "r") as f:
        print(f"行数: {len(f.readlines())}")

if __name__ == "__main__":
    count_lines("popular-names.txt")

# wcは行数をカウントするコマンド、-lは行数のみを表示するオプション
# wc -l popular-names.txt