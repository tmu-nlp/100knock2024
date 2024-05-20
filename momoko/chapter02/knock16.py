import numpy as np
num = int(input("ファイルの分割数を指定してください："))
basepath = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"

with open(basepath+"popular-names.txt", "r") as f:
    lines = f.read().splitlines()#文字列を改行位置で分割している
    file_len = len(lines)#2780
    num_list = range(file_len)#range(2780)->range(0,2780)を作ってくれる
    wakeru_num = np.array_split(num_list, num)#num_listをnumで分割．[0,1,2...num-1],[num,num+...
    #np.array_split：あまりが出ても，余り分を分割後の配列の先頭から振り分けていくイメージ．
    for i in range(num):
        out = open("knock16_{}.txt".format(i),"w")#分割したものの出力準備
        for j in wakeru_num:#[0,1,2...num-1],[num,num+...それぞれを指定．
            for mini_j in j:#jの中の数字を更に指定．
                out.write(lines[mini_j]+"\n")
        out.close()
f.close()