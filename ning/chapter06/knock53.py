import collections
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# 前処理関数
def Process(lines):
    sign_regrex = re.compile(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    lines = sign_regrex.sub("", lines)  # 記号を削除
    lines = re.sub(r"(\d+)", r" \1 ", lines)  # 数字と文字を分ける
    texts = lines.split()  # 空白で分割
    word_list = [word.lower() for word in texts]  # 小文字に変換
    return word_list

# ワンホットエンコーディング関数
def MakeOneHot(text, word_dict):
    word_list = Process(text)
    base_list = [0] * (len(word_dict) + 1)  # 長さを辞書のサイズ+1に設定
    for word in word_list:
        if word in word_dict:
            base_list[word_dict[word]] = 1
    return base_list

# 辞書の読み込み
def LoadDict(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    word_list = []
    for line in lines:
        word_list_temp = Process(line)
        word_list.extend(word_list_temp)
    c = collections.Counter(word_list).most_common()
    word_dict = {}
    i = 0
    for word in c:
        if word[1] > 1:
            word_dict[word[0]] = i
            i += 1
    return word_dict

# 記事見出しのカテゴリと予測確率を計算する関数
def PredictCategoryAndProbability(headline, model, word_dict):
    one_hot_vector = MakeOneHot(headline, word_dict)
    one_hot_vector_df = pd.DataFrame([one_hot_vector])
    prediction = model.predict(one_hot_vector_df)
    prediction_proba = model.predict_proba(one_hot_vector_df)
    return prediction[0], prediction_proba[0]

# 辞書を読み込み
GlobalWordDict = LoadDict("train.txt")

# 学習済みモデルの読み込み
model = joblib.load('logistic_regression_model.pkl')

# 記事見出しの例
headline = "Example headline for testing"

# 予測
category, probability = PredictCategoryAndProbability(headline, model, GlobalWordDict)
print(f"Predicted Category: {category}")
print(f"Prediction Probability: {probability}")
