text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
ls = text.split()
num = [1,5,6,7,8,9,15,16,19]
ans = {}

for i in range(len(ls)):
    if i+1 in num:
        ans[ls[i][0]] = i+1
    else:
        ans[ls[i][0:2]] = i+1

print(ans)