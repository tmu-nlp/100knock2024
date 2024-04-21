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