str1 = "パトカー"
str2 = "タクシー"
str3 = ""

n = len(str1) + len(str2)

for i in range(n):
  j = int(i / 2)

  if i % 2 == 0:
    str3 += str1[j]
  else:
    str3 += str2[j]

print(str3)
