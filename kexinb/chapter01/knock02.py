# task 02: 「パトカー」＋「タクシー」＝「パタトクカシーー」
# 「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．


def merge_alt_zip(str1, str2):
    result = ''
    for c1, c2 in zip(str1, str2):
        result += c1 + c2
    # appending leftovers
    shorterLen = min(len(str1), len(str2))
    result += str1[shorterLen:] + str2[shorterLen:]

    return result
    

if __name__ == "__main__":
    text1 = "パトカー1234"
    text2 = "タクシー56"

    result_zip = merge_alt_zip(text1, text2)

    print(result_zip)




# def merge_alt_loop(str1, str2):
#     i = 0
#     result = []
    
#     while i < len(str1) and i < len(str2):
#         result.append(str1[i])
#         result.append(str2[i])
#         i += 1
#     # appending leftovers
#     result.append(str1[i:])
#     result.append(str2[i:])
    
#     return "".join(result)