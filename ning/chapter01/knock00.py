s = "stressed"

#srl = list(reversed(s))
#reversedはlistしか使えないため、文字列をリスト化する

#sr = ''.join(list(reversed(s)))
#一文字ずつが要素として格納されたため、joinを使い一つの文字列に連結する

print(s[::-1])
