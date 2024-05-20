#各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
#各行の1列目の文字列の出現頻度を求め，その高い順に並べて表示せよ．確認にはcut, uniq, sortコマンドを用いよ．

zsr = df[0]
print(sr.value_counts())

#UNIXコマンド
!cut -f 1  popular-names.txt | sort | uniq -c | sort -n -r
