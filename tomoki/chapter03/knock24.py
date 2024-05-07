#No24(ファイル参照の抽出)
import pandas as pd
import re
df=pd.read_json("jawiki-country.json.gz",lines=True)
D_8=df[df["title"]=="イギリス"]
D_8=D_8["text"].values
#?を使うことで最短一致を可能にする
#\| メタ文字(|)自体の正規表現
#つまり｜が初めて見つかるまでの範囲を探す
# ex: ファイル:Wembley Stadium, illuminated.jpg|thumb|220px|[[ウェンブリー・スタジアム]]]]
#findall マッチしたものをリストにして返す
refe_file = re.findall("ファイル:(.+?)\|", D_8[0])
print("\n".join(refe_file))