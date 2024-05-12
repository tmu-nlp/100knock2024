def count_lines(file):
    try:
        with open(file, 'r') as file:
            lines = file.readlines()#ファイルの全ての行を読み込み
            return len(lines)#行のリスト=ファイルの行数
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    file = 'popular-names.txt'
    line_count = count_lines(file)
    if line_count is not None:
        print(f"行数: {line_count}")#行数がNoneでなかったら、その値を表示

#wc -l popular-names.txt  wcは行数を数える　-lは行数だけを表示する