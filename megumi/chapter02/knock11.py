#タブをスペースに置換
#タブ1文字につきスペース1文字に置換せよ．
#確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

with open("/Users/megumi/python_megumi/100knock2024/megumi/chapter02/popular-names.txt","r") as f:
    for line in f:
        print(line.replace('\t',' '))