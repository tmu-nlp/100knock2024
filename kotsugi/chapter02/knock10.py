count = 0

# with文：インデント中の処理が終了すると，自動でファイルクローズする． 
with open("./woodnx/chapter02/popular-names.txt", "r") as f:
  for line in f:
    count += 1

print(f'count: {count}')

# wc -l './woodnx/chapter02/popular-names.txt'

###  pythonコードwlコマンドでカウント数が異なる理由 ###
# wlコマンドは行数をカウントする際，改行コード（\n）数を数えており，
# 該当ファイルは最終行に改行がないため，pythonコードより1少ない値になる．
