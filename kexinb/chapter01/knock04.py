# task04: 元素記号
# “Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

import string

rawText = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
strippedText = rawText.translate(str.maketrans("","",string.punctuation))
wordList = strippedText.split()

indList = [1, 5, 6, 7, 8, 9, 15, 16, 19]


resultDict = dict(enumerate(wordList, 1))  ### enumerate(iterable, start=0)
for i in resultDict:
    if i in indList:
        resultDict[i] = resultDict[i][:1]
    else:
        resultDict[i] = resultDict[i][:2]
print(resultDict)