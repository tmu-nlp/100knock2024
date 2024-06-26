#“Hi He Lied Because Boron Could Not Oxidize Fluorine. 
#New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，
#1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字
#それ以外の単語は先頭の2文字を取り出し
#取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．


text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
dic = {}
words = text.split()

for i in range(len(words)):
    if i+1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]:#i+1に注意
        word = words[i][0]
    else:
        word = words[i][:2]
    dic[word] = i + 1#循環
print(dic)