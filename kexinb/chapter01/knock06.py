# task06: 集合
# “paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

from knock05 import n_gram

# list1 = n_gram(list("paraparaparadise"), 2)
# list2 = n_gram(list("paragraph"), 2)

# set(list1)

text1 = 'paraparaparadise'
text2 = 'paragraph'
bigramSet1 = set(n_gram(text1, 2))
bigramSet2 = set(n_gram(text2, 2))

print(f"Union: {bigramSet1 | bigramSet2}")
print(f"Intersection: {bigramSet1 & bigramSet2}")
print(f"Difference: {bigramSet1 - bigramSet2}")
print(f"Is 'se' in both sets: {'se' in bigramSet1 & bigramSet2}")