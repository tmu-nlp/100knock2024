"""
特徴量抽出
学習データ，検証データ，評価データから特徴量を抽出し，
それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ．
なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう
"""

"""
TfidfVectorizerを使うことによってテキストデータを数値ベクトルに変換する
TF（Term Frequency）：ある文書内で特定の単語が出現する頻度。
TF(t,d)=その単語の出現回数/文書内のすべての単語の出現回数
IDF（Inverse Document Frequency）：特定の単語が文書集合全体でどれだけ重要かを示す尺度。頻繁に出現する単語には低い値が割り当てられる。
IDF(t)=log(文書の総数/その単語が出現する文書の数)
TF-IDF(t,d)=TF(t,d)xIDF(t)
"""
import collections
import re
import pandas as pd

# 前処理関数
def Process(lines):
    sign_regrex = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    lines = sign_regrex.sub("", lines)  # 記号を削除
    lines = re.sub(r"(\d+)", r" \1 ", lines)  # 数字と文字を分ける
    texts = lines.split()  # 空白で分割
    word_list = [word.lower() for word in texts]  # 小文字に変換
    return word_list

# 辞書作成関数
def MakeDict(name):
    with open(f"{name}.txt", "r") as f:
        lines = f.readlines()
    word_list = []
    for line in lines:
        word_list_temp = Process(line)
        word_list.extend(word_list_temp)
    c = collections.Counter(word_list).most_common()
    word_dic = {}
    i = 0  # インデックスを0から始める
    for word in c:
        if word[1] > 1:
            word_dic[word[0]] = i
            i += 1
    return word_dic

# ワンホットエンコーディング関数
def MakeOneHot(text):
    word_list = Process(text)
    base_list = [0] * (len(GlobalWordDict) + 1)  # 長さを辞書のサイズ+1に設定
    for word in word_list:
        if word in GlobalWordDict:
            base_list[GlobalWordDict[word]] = 1
    return base_list

# 特徴量抽出関数
def MakeFeatureText(name):
    df = pd.read_csv(f"{name}.txt", sep='\t', header=None, names=['Category', 'Title'])
    df_2 = pd.DataFrame([MakeOneHot(title) for title in df["Title"]])
    df_3 = pd.concat([df, df_2], axis=1)
    df_3.to_csv(f"{name}.feature.txt", sep='\t', index=False)

# 辞書をグローバル変数で定義
GlobalWordDict = MakeDict("train")

# 特徴量抽出とファイル保存
MakeFeatureText("train")
MakeFeatureText("test")
MakeFeatureText("valid")
