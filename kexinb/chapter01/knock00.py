# task 00: 文字列の逆順
# 文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．

def reverse(s):
    return s[::-1]

if __name__ == "__main__":
    text = "stressed"
    print (reverse(text))