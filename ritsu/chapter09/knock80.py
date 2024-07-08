import pandas as pd
from collections import defaultdict
import os

def load_data(file_path):
    """データファイルを読み込む関数"""
    return pd.read_csv(file_path, sep='\t', names=['CATEGORY', 'TITLE'])

def create_word_to_id(train_data):
    """単語をID番号に変換する辞書を作成する関数"""
    # 単語の出現回数をカウントする辞書, defaultdictを使うと初期値が0
    word_count = defaultdict(int)
    
    # 単語の出現回数をカウント
    for title in train_data['TITLE']:
        for word in title.split():
            word_count[word] += 1
    
    # 出現回数でソートし、ID番号を付与
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    word_to_id = {}
    for i, (word, count) in enumerate(sorted_words):
        if count < 2:
            break
        word_to_id[word] = i + 1
    
    return word_to_id

def convert_to_ids(text, word_to_id):
    """与えられた文章をID番号のリストに変換する関数"""
    return [word_to_id.get(word, 0) for word in text.split()]

def main():
    # データの読み込み
    base_path = os.path.join('..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    valid_data = load_data(os.path.join(base_path, 'valid.txt'))
    test_data = load_data(os.path.join(base_path, 'test.txt'))

    # 単語をID番号に変換する辞書を作成
    word_to_id = create_word_to_id(train_data)

    # 検証用データの最初の文章を変換
    sample_text = valid_data['TITLE'].iloc[0]
    sample_ids = convert_to_ids(sample_text, word_to_id)

    # 結果の表示
    print("元の文章:")
    print(sample_text)
    print("\nID番号に変換された文章:")
    print(sample_ids)

    # 単語とIDの対応関係を表示（上位10件）
    print("\n単語とIDの対応関係（上位10件）:")
    for word, id_ in list(word_to_id.items())[:10]:
        print(f"{word}: {id_}")

if __name__ == "__main__":
    main()

"""
元の文章:
PRECIOUS-Gold ends flat as S&P 500 rises; platinum up

ID番号に変換された文章:
[192, 327, 303, 5, 185, 279, 0, 3830, 33]

単語とIDの対応関係（上位10件）:
to: 1
...: 2
in: 3
on: 4
as: 5
UPDATE: 6
-: 7
for: 8
of: 9
The: 10
"""