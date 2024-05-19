#問11
#タブ1文字につきスペース1文字に置換せよ．
#確認にはsedコマンド，trコマンド，もしくはexpandコマンドを用いよ．

#with構文：open()を用いた後、自動的にcloseしてくれる

import pandas as pd
import subprocess

with open("/home/mohasi/HelloWorld/100knock2024/mio/chapter02/popular-names.txt") as f :

#データを一行ずつ読み込む→改行を削除→タブ1文字につきスペース1文字に置換
    for data in f:
        data = data.replace("\n", "")
        print(data.replace("\t", " "))

#UNIXコマンド
#sedコマンド：入力を行単位で読み取り、テキスト変換などの編集をおこない行単位で出力する。
#             正規表現に対応している https://tech-blog.rakus.co.jp/entry/20211022/sed　

#　　　　　　 オプション[-e（スクリプト）]：スクリプトをコマンドに追加(複数指定すれば複数回のコマンドを実行可能）
#コマンド[s/置換前文字列/置換後文字列/]：置換前文字列で指定した文字列にマッチした部分を置換後文字列へ置き換え
# 複数マッチ→先頭のみ置換/対象全て置換→［-g］オプション併用（例：［s/置換前/置換後/g］）
command = ["sed", "-e 's/\t/ /g' ./popular-names.txt"]
subprocess.check_output(command) 