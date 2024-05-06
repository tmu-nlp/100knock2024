#https://ja.wikipedia.org/wiki/Help:%E6%97%A9%E8%A6%8B%E8%A1%A8
#によると，=が2~6個に挟まれている

import re
basepath="/Users/shirakawamomoko/Desktop/nlp100保存/chapter03/"
with open(basepath+"uk_articles.csv","r") as f:
    lines = f.readlines()
f.close()

pat = "\={2,6}.*?\={2,6}"#=が2~6個あって，挟まれている
for l in lines:
    if re.match(pat,l):
        moji = l.replace("\n","").replace("=","")#セクション名のみ抽出
        kazu = int(l.count("=")/2 -1)
        print(moji,kazu)