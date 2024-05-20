path = "/Users/shirakawamomoko/Desktop/nlp100保存/chapter02/"
with open(path+"ノック13.txt","w") as w:
    with open(path+"col1.txt","r") as one,open(path+"col2.txt","r") as two:
        for i in range(2780):
            name = one.readline()#oneを1行ずつ
            sex = two.readline()#twoを1行ずつ
            count = 0
            d = "\t".join((name,sex))
            if count != 2780:
                w.write(d+"\n")
                count += 1
            else:
                w.write(d)
    one.close()
    two.close()
w.close()