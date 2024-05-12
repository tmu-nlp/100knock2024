# task11. タブをスペースに置換
# タブ1文字につきスペース1文字に置換せよ．確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

import filecmp

def tabs_to_spaces(text):
    return text.replace("\t", " ")

if __name__ == '__main__':
    filepath_in = 'data/popular-names.txt'
    filepath_out = filepath_in.replace('popular-names.txt', 'popular-names-spaces.txt')

    with open(filepath_in, 'r') as f:
        input = f.read()
        output = tabs_to_spaces(input)

    with open(filepath_out, 'w') as f:
        f.write(output)

    # assert filecmp.cmp(filepath_out, "data/popular-names-test.txt")