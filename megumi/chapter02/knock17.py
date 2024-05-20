#各行を3コラム目の数値の降順にソート
#各行を3コラム目の数値の逆順で整列せよ
#（注意: 各行の内容は変更せずに並び替えよ）．
#確認にはsortコマンドを用いよ
#（この問題はコマンドで実行した時の結果と合わなくてもよい）．

df.sort_values(2, ascending=False)

#UNIXコマンドで確認
#UNIXコマンド
!cut -f 3  popular-names.txt | sort -n -r
