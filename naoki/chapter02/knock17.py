import urllib
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    name_list = []
    lines = [line.decode('utf-8') for line in f.readlines()]
    #これで列ごとに処理してくれる
    for line in lines:
        #スプリットで各行をタブで分割
        #参考list = ['母\\t', '父\\t']
        # family = []
        # for c in list:
        ##タブ文字'\\t'で分割し、最初の要素をfamilyリストに追加
        #     family.append(c.split('\\t')[0])
        name_list.append(line.split('\t')[0])
#print(name_list)   