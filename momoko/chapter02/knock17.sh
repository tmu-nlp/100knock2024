cut -d " " -f 1 "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02//popular-names.txt"|sort|uniq
#-d：区切り文字について，タブの代わりに使用する文字を指定する．今回は" "．
#-f 1：1列目を指定
#sort：辞書順にデータを並び替え
#uniq：重複データの削除(連続しているものを．sortで並び替えしている．)
