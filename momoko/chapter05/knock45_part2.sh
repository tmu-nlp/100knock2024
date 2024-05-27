cat "knock45.txt" |grep "行う"|sort|uniq -c|sort -nr
#5 行う               て
#1 行う               が      という
#1 行う               と      も
#1 行う               て      と
#1 行う               も
#1 行う               が

cat "knock45.txt" |grep "なる"|sort|uniq -c|sort -nr
#5 なる               て
#1 異なる             に
#1 なる               により
#1 なる               か      と
cat "knock45.txt" |grep "与える"|sort|uniq -c|sort -nr
#   1 与える             て
