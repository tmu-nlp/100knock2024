# task20. JSONデータの読み込み
# Wikipedia記事のJSONファイルを読み込み，「イギリス」に関する記事本文を表示せよ．問題21-29では，ここで抽出した記事本文に対して実行せよ．

import json, gzip

def load_uk_data(fname):
    with gzip.open(fname, "r") as f: 
        for line in f: 
            json_data = json.loads(line)
            if ("title","イギリス") in json_data.items():
                return json_data["text"]

if __name__ == "__main__":
    print(load_uk_data("chapter03/jawiki-country.json.gz"))

# import pandas as pd

# def load_uk_data(path):
#     df = pd.read_json('chapter03/jawiki-country.json.gz', lines=True)
#     uk_data = df.query('title=="イギリス"')['text'].values[0]
#     return uk_data

# if __name__ == "__main__":
#     print(load_uk_data('chapter03/jawiki-country.json.gz'))
