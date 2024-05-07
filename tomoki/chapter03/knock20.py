#No20(JSON(JavaScriptデータにおけるオブジェクトの書き方を参考に作られたデータ)の読み込み)
import pandas as pd
from IPython.display import display
#.gzでzip形式のままで読み込める
#JSON Lines形式(改行区切りで1行1JSONオブジェクトという形式で定義されたファイル)はkines=Trueにする。
df=pd.read_json("jawiki-country.json.gz",lines=True)
#titleが「イギリス」の行をdf化する
D_8=df[df["title"]=="イギリス"]

#dfは「values」「columns」「index」の３つの要素からなる。その内のvaluesを取り出してdf化
D_8=D_8["text"].values
display(D_8)





