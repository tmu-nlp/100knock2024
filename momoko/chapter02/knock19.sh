awk '{print $1}' "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/popular-names.txt" | sort | uniq -c | sort -rn
#awk '{print $1}'：各行の名前を抽出する．
#sort：名前をアルファベット順にする．
#uniq -c：重複削除，出現回数の集計．
#sort -rn：出現回数を降順にソート．