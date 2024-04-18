# task04: 元素記号
# “Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

import string

def strip_puncs(raw):
    ### translate(translation table) 
    ### maketrans(chars to be replaced, their replacements, chars to be deleted) -> translation table
    return raw.translate(str.maketrans('', '', string.punctuation))

def get_initials(raw, indices):
    wordList = (strip_puncs(raw)).split()
    resultDict = dict(enumerate(wordList, 1))  ### enumerate(iterable, start=0)
    for i in resultDict:
        if i in indices:
            resultDict[i] = resultDict[i][:1]
        else:
            resultDict[i] = resultDict[i][:2]
    return resultDict

if __name__ == "__main__":
    rawText = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    indList = [1, 5, 6, 7, 8, 9, 15, 16, 19]
    print(get_initials(rawText, indList))