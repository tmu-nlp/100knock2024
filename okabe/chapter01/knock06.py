def n_gram_devider(sequence, n):
    letter_n_gram = []
    for i in range(len(sequence)-n+1):
        letter_n_gram.append(sequence[i:i+n])

    return letter_n_gram

seq_1 = 'paraparaparadise'
seq_2 = 'paragraph'
X = n_gram_devider(seq_1,2)
Y = n_gram_devider(seq_2,2)

union = []
intsec = []
dif = []

temp_union = X + Y
union = list(set(temp_union))

intsec_temp = []
for elm in X:
    if elm in Y:
        intsec_temp.append(elm)
intsec = list(set(intsec_temp))

for elm in union:
    if elm not in intsec:
        dif.append(elm)

print("union:", union, "intsec:", intsec, "dif:", dif)

if 'se' in X:
    print('se in X')
else :
    print('se not in X')
if 'se' in Y:
    print('se in Y')
else:
    print('se not in Y')