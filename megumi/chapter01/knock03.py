#03.円周率
text03='Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
#句読点を除去
text03=text03.replace(",","").replace(".","")
#単語を分割
text03=text03.split()
print([len (i) for i in text03])