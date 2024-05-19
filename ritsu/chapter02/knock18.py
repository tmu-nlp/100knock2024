def sort_by_third_column(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    # 3列目の数値に基づいて各行を降順にソート
    lines.sort(key=lambda x: float(x.split('\t')[2]), reverse=True) # keyでソートの基準を指定, lambdaで各行の3列目を取得, xは各行, floatで数値に変換
    return lines

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用法: python knock18.py ファイル名")
    else:
        filename = sys.argv[1]
        sorted_lines = sort_by_third_column(filename)
        for line in sorted_lines:
            print(line.strip())  # 行の末尾の改行を削除して出力

# # ファイルを3列目の数値に基づいて降順にソート, -kでソートの基準を指定, 3,3で3列目を指定, nで数値としてソート, rで降順
# sort -k 3,3nr popular-names.txt
