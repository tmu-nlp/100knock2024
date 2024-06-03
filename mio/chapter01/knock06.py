#06. 集合
#06 “paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．
#さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．
sequence_X = "paraparaparadise"
sequence_Y = "paragraph"
n=2
X  = set(n_gram(n, sequence_X))
Y = set(n_gram(n, sequence_Y))

#参考
print("※参考:集合X:{}".format(X))
print("※参考:集合Y:{}".format(Y))

#和集合
print("和集合:{}".format(X.union(Y)))

#積集合
print("積集合:{}".format(X.intersection(Y)))

#差集合
print("差集合:{}".format(X.difference(Y)))

#’se’というbi-gramがXに含まれるか
print("'se'というbi-gramがXに含まれるか:{}".format("se" in X))
#’se’というbi-gramがYに含まれるか
print("'se’というbi-gramがYに含まれるか:{}".format("se" in Y))