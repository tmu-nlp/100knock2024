from collections import Counter

def count_first_column(filename):
    with open(filename, 'r') as file:
        first_column = [line.split('\t')[0] for line in file]  # 1列目のデータをリストとして抽出
    return Counter(first_column)  # Counterで出現回数をカウント, 型は辞書型

def print_sorted_counts(filename):
    counts = count_first_column(filename)
    # 出現頻度で降順にソートし、それを表示
    for item, count in counts.most_common(): # most_commonで出現頻度の高い順に取得
        print(f"{count}: {item}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用法: python knock19.py popular-names.txt")
    else:
        filename = sys.argv[1]
        print_sorted_counts(filename)

# # -f1で1列目を抽出, sortでソート, uniq -cで重複を排除し出現回数をカウント, sort -nrで出現回数で降順にソート
# cut -f1 filename.txt | sort | uniq -c | sort -nr
