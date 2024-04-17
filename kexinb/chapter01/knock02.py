# task 02: 「パトカー」＋「タクシー」＝「パタトクカシーー」
# 「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

text1 = "パトカー1234"
text2 = "タクシー56"

##### Method 1: conventional loop
i = 0
result = []

while i < len(text1) and i < len(text2):
    result.append(text1[i])
    result.append(text2[i])
    i += 1
# assume appending leftovers at the end?
result.append(text1[i:])
result.append(text2[i:])

print("".join(result))