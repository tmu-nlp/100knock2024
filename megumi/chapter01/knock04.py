#04.元素記号
#“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”
#という文を単語に分解し，1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．
text04='Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
text04=text04.replace(".","")
text04=text04.split()

num=[1,5,7,8,9,15,16,19]
dict={} #空の辞書用意

for i,j in enumerate(text04): #enumerate()で各文字のインデックスと要素を取得
    if i+1 in num: #保存してある番号と各文字列のインデックスインデックス＋１を比較して、同じであれば一文字、それ以外は２文字に変換し、インデックス＋1を要素にして辞書に追加
        dict[j[0]]=i+1
    else:
        dict[j[:2]]=i+1
print(dict)