import urllib.request
# try:
#     with urllib.request.urlopen("https://nlp100.github.io/data/popular-names.txt") as f:
#         name_list = [line.decode('utf-8').split('\t')[0] for line in f]
# except urllib.error.URLError as e:
#     print(f"URLエラー: {e.reason}")
# except urllib.error.HTTPError as e:
#     print(f"HTTPエラー: {e.code} - {e.reason}")
# except Exception as e:
#     print(f"予期せぬエラー: {e}")
def sort_key(line):
    return int(line.decode('utf-8').split('\t')[2])

try:
    with urllib.request.urlopen("https://nlp100.github.io/data/popular-names.txt") as f:
        lines = f.readlines()
        lines.sort(key=sort_key, reverse=True)
        sorted_lines = [line.decode('utf-8') for line in lines]
except Exception as e:
    print(f"エラーが発生しました: {e}")