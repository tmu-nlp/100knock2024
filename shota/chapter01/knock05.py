def n_gram(text,n):
    ls = []
    for i in range(0,len(text)-n+1):
        ls.append(text[i:i+n])
    return ls

text = "I am an NLPer"
text_ = text.split()

print(text_)

print(n_gram(text,2))
print(n_gram(text_,2))