#斜体~斜体と強調のこと...??
#'で囲われているやつっていう解釈であってる...??
import re
basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
with open(basepath+"uk_articles.csv","r") as f:
    lines = f.readlines()
f.close()

patee = "\|.*?\=.*?\n"#基礎情報の書き方例：|略名  =イギリス
kiso_n=[]
kiso_z=[]
for l in lines:
    if re.match(patee,l):
        kiso = l.replace("|","").replace(" ","").replace("'","").replace("\n","").split("=")#.replaceを’についても追加
        kiso_n.append(kiso[0])
        kiso_z.append(kiso[1])
        if kiso[0]=="注記":#基礎情報の最後が注記らしいので，それが来たらbreak
            break

kiso_dict = dict(zip(kiso_n,kiso_z))
print(kiso_dict)