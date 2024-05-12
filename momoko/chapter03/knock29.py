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
        kiso = l.replace("|","").replace("'","").replace("\n","")
        kiso = kiso.replace("[","").replace("]","").split("=")#内部リンクは[[]]で囲われているので，それを消す．
        kiso_n.append(kiso[0].replace(" ",""))
        kiso_z.append(kiso[1])
        if kiso[0]=="注記":#基礎情報の最後が注記らしいので，それが来たらbreak
            break

kiso_dict = dict(zip(kiso_n,kiso_z))

#ここからknock29
import requests
uk_flag = kiso_dict["国旗画像"]#Flag of the United Kingdom.svg
uk_flag = uk_flag.replace(" ", "_")#Flag_of_the_United_Kingdom.svg
S = requests.Session()#ログインが必要なサイトを覗く．ログインを再利用する，みたい，な...??
url = "https://www.mediawiki.org/w/api.php"

PARAMS = {#URLに「?key=value」の形で付加するパラメーターを指定することができる．
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "iiprop":"url",
    "titles": "File:"+uk_flag
}

R = S.get(url, params=PARAMS)#webページの取得
DATA = R.json()#requestsオブジェクトが辞書の形に変換される
uk_flag_url = DATA["query"]["pages"]["-1"]["imageinfo"][0]["url"]#辞書から該当部分を参照する
print(uk_flag_url)