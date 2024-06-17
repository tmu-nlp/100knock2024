#30.形態素解析結果の読み込み
"""
形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をキーとするマッピング型に格納し，
1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．
"""

# 関数の定義
def parse_neko():
    result = []
    sentence = []

    # 形態素解析済みのファイルを開き、1行ずつ読み込む
    with open("neko.txt.mecab") as f:
        for line in f:
            # 1行をタブでsplit()する（要素が2つのlistが返ってくる）
            l1 = line.split("\t")
            # l1の要素が2つであれば、l1[1]の要素をカンマで分割する
            if len(l1) == 2:
                l2 = l1[1].split(",")
                # 問題の通り、4つのキーを指定して、dictを作成し、sentenceに追加していく
                sentence.append({"surface": l1[0], "base": l2[6], "pos": l2[0], "pos1": l2[1]})
                # 句点（。）が来たときに、sentence内のdictをresultに追加する
                if l2[1] == "句点":
                    result.append(sentence)
                    sentence = []

    return result

# 関数の呼び出し
result = parse_neko()
print(result)
"""
出力結果
 {'surface': 'ある', 'base': 'ある', 'pos': '動詞', 'pos1': '自立'},
 {'surface': 'から', 'base': 'から', 'pos': '助詞', 'pos1': '接続助詞'}, 
 {'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'},
"""