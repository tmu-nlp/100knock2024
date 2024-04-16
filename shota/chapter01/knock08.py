text = input()

def ciper(text):
    cipered = ""
    for c in text:
        asc = ord(c)
        if 97 <= asc <= 122:
            c = chr(219-asc)
        cipered += c
    return cipered

def deciper(text):
    decipered = ""
    for c in text:
        asc = ord(c)
        if 97 <= asc <= 122:
            c = chr(219-asc)
        decipered += c
    return decipered

cipered_text = ciper(text)
print(" cipered_text  >> ",cipered_text)

decipered_text = deciper(cipered_text)
print("decipered_text >> ",decipered_text)