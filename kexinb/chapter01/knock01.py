# task 01: パタトクカシーー
#「パタトクカシーー」という文字列の1,3,5,7文字目を取り出して連結した文字列を得よ．

def intermittent(s):
    return s[::2]

if __name__ == "__main__":
    text = "パタトクカシーー"
    print (intermittent(text))