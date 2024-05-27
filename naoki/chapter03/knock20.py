import json
import gzip
import re

text_list = []
with gzip.open('C:/Users/shish_sf301y1/Desktop/pyファイル/jawiki-country.json.gz') as f:
    lines = f.readlines()
    for line in lines:
        text_list.append(json.loads(line)) 
        #ここが分からない,line[i]はbyte型では？

for i in range(len(text_list)):
    if text_list[i]['title']=="イギリス":
        UK_text = str(text_list[i])   