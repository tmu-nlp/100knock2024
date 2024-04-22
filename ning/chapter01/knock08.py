import re
def cipher(text):
    repatter = re.compile('[a-z]')
    temp = ""
    for ch in text:
        if re.match(repatter, ch):
            ch = chr(219 - ord(ch))
        temp += ch
    return temp

em = cipher("hello world!")
print("暗号化:",em)
dm = cipher(em)
print("復号化:",dm)

