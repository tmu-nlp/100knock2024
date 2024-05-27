def get_unique_names(filename):
    unique_names = set() # setで集合を作成し重複を取り除く
    with open(filename, 'r') as file:
        for line in file:
            columns = line.split('\t')  # タブで分割
            unique_names.add(columns[0])  # 1列目の値をセットに追加
    return unique_names

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用法: python nock17.py ファイル名")
    else:
        filename = sys.argv[1]
        unique_names = get_unique_names(filename)
        sorted_unique_names = sorted(unique_names)
        for name in sorted_unique_names:
            print(name)

# # 1列目のデータを抽出し、ソートして重複を排除する
# cut -f 1 popular-names.txt | sort | uniq
