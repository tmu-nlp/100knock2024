#No19(各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる)
import pandas as pd
df = pd.read_table('popular-names.txt', header=None)
fre= df[0]
#value_counts(ユニークな要素の頻度を数える)
print(fre.value_counts())

#-c(重複した行数も出力)
#cut -f 1  popular-names.txt | sort | uniq -c | sort -n -r

