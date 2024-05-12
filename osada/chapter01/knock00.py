
## 問題0　文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ

text1="stressed"
temp=""
for i in reversed(text1):
  temp+= i
print(temp)  
