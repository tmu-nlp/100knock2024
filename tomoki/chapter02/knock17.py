#No17(１列目の文字列の異なり)
import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
#unixコマンドと同じように処理する（但しunixコマンドではsortしてからuniqを使う)
dif= df[0].unique()
dif.sort()
print (dif)

#-f　数字によって、切り出すフィールド数を指定する。
#cut -f 1  popular-names.txt | sort | uniq