import sys

def display_tail(filename, n):
    """指定されたファイルから末尾のn行を表示する"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()  # ファイルの全行を読み込む
            for line in lines[-n:]:  # 末尾からn行を取得して表示
                print(line.strip())
    except FileNotFoundError:
        print("指定されたファイルが見つかりません")
    except ValueError:
        print("行数は正の整数である必要があります")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用法: python knock15.py ファイル名 行数")
    else:
        filename = sys.argv[1]
        try:
            n = int(sys.argv[2])
            if n > 0:
                display_tail(filename, n)
            else:
                print("行数は正の整数である必要があります")
        except ValueError:
            print("行数は整数で入力してください")


# # 指定されたファイルの末尾からN行を表示する
# tail -n N popular-names.txt
