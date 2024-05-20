#NO11(タブをスペースに変換)
with open("popular-names.txt") as f:
    for i in f:
        print(i.replace("\t"," "))

#sed 's/\t/ /g' popular-names.txt > replaced.txt
#sed s/置換対象文字列/置換後文字列/g' ファイル名
