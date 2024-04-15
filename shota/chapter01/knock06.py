text1 = "paraparaparadise"
text2 = "paragraph"

def n_gram(text,n):
    ls = []
    for i in range(0,len(text)-n+1):
        ls.append(text[i:i+n])
    return ls

X = set(n_gram(text1,2))
Y = set(n_gram(text2,2))

print("union >>",X | Y)
print("intersection >>",X & Y)
print("difference >>",X - Y)

if "se" in X:
    print("'se' is in X")
else:
    print("'se' is not in X")

if "se" in Y:
    print("'se' is in Y")
else:
    print("'se' is not in Y")