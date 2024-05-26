import sys # コマンドライン引数を取得するためのライブラリ

def display_head(filename, n):
    """指定されたファイルから先頭のn行を表示する"""
    try:
        with open(filename, 'r') as file:
            for i in range(n):
                print(file.readline().strip())
    except FileNotFoundError:
        print("指定されたファイルが見つかりません")
    except ValueError:
        print("行数は正の整数である必要があります")

if __name__ == "__main__":
    if len(sys.argv) != 3: # コマンドライン引数が3つでない場合
        print("使用法: python knock14.py ファイル名 行数")
    else: 
        filename = sys.argv[1]
        try: 
            n = int(sys.argv[2]) # 行数を整数に変換
            if n > 0: # 行数が正の整数の場合
                display_head(filename, n) # ファイルの先頭n行を表示
            else:
                print("行数は正の整数である必要があります")
        except ValueError:
            print("行数は整数で入力してください")

# # 指定されたファイルの先頭からN行を表示する
# head -n N popular-names.txt
