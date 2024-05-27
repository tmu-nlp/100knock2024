#07. テンプレートによる文生成
#07 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．
#さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def temperature(x, y, z):
  print("{}時の".format(x) + y +"は{} ".format(z))
x=12
y="気温"
z=22.4
temperature(x, y, z)