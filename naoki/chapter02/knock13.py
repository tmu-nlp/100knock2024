import urllib.request
with open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col1.txt") as f1 , open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col2.txt") as f2 , open("C:/Users/shish_sf301y1/Desktop/pyファイル/output_col1,2.txt","w") as f12:
    name = f1.readlines()
    sex = f2.readlines()
    for n,s in zip(name,sex):
        n = n.replace('\n','')
        s = s.replace('\n','')
        f12.write(n+'\t'+s)