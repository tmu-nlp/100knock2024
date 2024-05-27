import sys
import math

def split_file(filename, n):
    """ファイルをn分割する"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()  # ファイルの全行を読み込む

        total_lines = len(lines)
        part_size = math.ceil(total_lines / n)  # 1つのファイルに入る行数を計算, math.ceilは切り上げ
        for i in range(n):
            start_index = i * part_size
            if start_index >= total_lines:  # 開始インデックスがファイルの総行数を超えていたら終了
                break
            end_index = min((i + 1) * part_size, total_lines)  # 範囲を超えないように調整
            part_lines = lines[start_index:end_index]  # 分割部分を取得
            with open(f'part_{i + 1}.txt', 'w') as part_file:
                part_file.writelines(part_lines)  # 分割ファイルに書き込む
    except FileNotFoundError:
        print("指定されたファイルが見つかりません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用法: python knock16.py ファイル名 分割数")
    else:
        filename = sys.argv[1]
        try:
            n = int(sys.argv[2])
            if n > 0:
                split_file(filename, n)
            else:
                print("分割数は正の整数である必要があります")
        except ValueError:
            print("分割数は整数で入力してください")

# # ファイルの総行数を計算
# TOTAL_LINES=$(wc -l < popular-names.txt)

# # 必要な行数を計算（Pythonのmath.ceilに相当）
# LINES_PER_FILE=$(( ($TOTAL_LINES + n - 1) / n ))

# # ファイルを指定された行数で分割し、分割されたファイルに.txt拡張子を付けて保存する
# split -l $LINES_PER_FILE --numeric-suffixes=1 --additional-suffix=.txt popular-names.txt part_u