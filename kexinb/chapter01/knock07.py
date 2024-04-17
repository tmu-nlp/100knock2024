# task07: テンプレートによる文生成
# 引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def template_text_gen(time, attr, val):
    return f'{time}時の{attr}は{val}.'


x = 12
y = '気温'
z = 22.4
print(template_text_gen(x, y, z))