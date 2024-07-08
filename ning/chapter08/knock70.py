from gensim.models import KeyedVectors
import pandas as pd
import re
import torch

# カテゴリをラベルに変換する関数
def EncoderNN(sign):
    if sign == "b":
        code = 0
    elif sign == "t":
        code = 1
    elif sign == "e":
        code = 2
    elif sign == "m":
        code = 3
    else:
        raise ValueError("Invalid category sign")
    return code

# テキストをベクトルに変換する関数
def Text2Vec(text, model):
    words = text.split(" ")
    vec_sum = 0
    length = 0
    for word in words:
        try:
            temp = model.get_vector(word)
            vec_sum += temp
            length += 1
        except KeyError:
            continue
    return vec_sum / length if length != 0 else vec_sum

# データを読み込み、変換する関数
def TorchData(data, model, path):
    try:
        df = pd.read_table(f"{path}/{data}.txt")
    except Exception as e:
        print(f"Error reading {data}.txt: {e}")
        return
    
    # 特殊文字を削除するための正規表現
    sign_regrex = re.compile('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`|＄＃＠£â€™]')
    f_regrex = lambda x: sign_regrex.sub("", x)
    df["TITLE"] = df["TITLE"].map(f_regrex)
    
    # タイトルをベクトルに変換
    X_torch = torch.tensor([Text2Vec(title, model) for title in df["TITLE"]], dtype=torch.float32)
    torch.save(X_torch, f"{path}/X_{data}.pt")
    
    # カテゴリをラベルに変換
    df["CATEGORY"] = df["CATEGORY"].map(EncoderNN)
    Y_torch = torch.tensor(df["CATEGORY"].values, dtype=torch.long)
    torch.save(Y_torch, f"{path}/Y_{data}.pt")

# モデルのロード
model_path = "GoogleNews-vectors-negative300.bin.gz"
try:
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# データの処理
data_path = "chapter08"
TorchData("train", model, data_path)
TorchData("test", model, data_path)
TorchData("valid", model, data_path)
