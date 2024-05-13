def count_lines(filename):
    with open(filename, 'r') as f:
        count = sum(1 for line in f)  # １の数をカウント
    return count

print(count_lines('popular-names.txt'))
